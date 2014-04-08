#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <openssl/sha.h>

uint32_t find_vanity(uint8_t* vanity, int vlen, uint8_t* key, int keylen) {

    time_t start = time(NULL);

    // 5 to 8 are the timestamp (the interesting bit!)
    uint32_t timestamp = (key[4] << 24) + (key[5] << 16) + (key[6] << 8) + key[7];

    // maybe make this actually variable...
    unsigned int limit = 1262300400;
    unsigned int total = timestamp-limit;

    {
        char buf[200];
        time_t ye = limit;
        struct tm* tmp = localtime(&ye);
        strftime(buf, sizeof(buf), "%F", tmp);
        printf("[*] Searching down to %s\n", buf);
    }

    unsigned int counter = 0;
    while(timestamp > limit) {
        // reduce by one
        timestamp -= 1;
        key[4] = (timestamp >> 24) & 0xff;
        key[5] = (timestamp >> 16) & 0xff;
        key[6] = (timestamp >> 8) & 0xff;
        key[7] = (timestamp) & 0xff;

        // calculate hash with new data
        uint8_t digest[20];
        SHA1(key, keylen, digest);

        // see if it's a vanity one
        int i;
        for(i = 0; i < vlen; i++) {
            if(digest[19-i] != vanity[vlen-1-i])
                break;
            // if we're here, we found one! yay!
            if(i == vlen-1) {
                printf("\n");
                return timestamp;
            }
        }
        if((++counter % 10000000) == 0) {
            char buf[200];
            time_t diff = time(NULL) - start +1;

            time_t ye = timestamp;
            struct tm* tmp = localtime(&ye);
            strftime(buf, sizeof(buf), "%F", tmp);
            printf("[~] At %s, %d%% at %u kps\n", buf, (int)(((double) counter)/(total)*100), counter / diff);
        }
    }

    return 0;

}

void readkey(int fd, uint8_t** key, int* keylen, uint8_t** uid, int* uidlen) {

    uint8_t buf[3];
    {
        // make sure the first packet is a pubkey or seckey packet
        read(fd, buf, 3);
        if((buf[0] & 0x3f) >> 2 != 6 && (buf[0] & 0x3f) >> 2 != 5) {
            printf("[!] packet is not a pubkey or seckey!\n");
            exit(2);
        }

        *keylen = (buf[1] << 8) + (buf[2]);
        // take into account the three bytes already read
        *keylen += 3;

        // assuming we work with 4096 keys at most
        *key = (uint8_t*) malloc(sizeof(uint8_t)**keylen);
        // copy first three bytes from buf
        memcpy(*key, buf, 3);
        // read rest of the key
        read(fd, *key+3, (*keylen)-3);

        if((*key)[3] != 0x04) {
            printf("[!] version number != 4\n");
            exit(2);
        }
    }

    // the next packet must be a uid packet
    {
        // read uid packet
        read(fd, buf, 2);
        if((buf[0] & 0x3f) >> 2 != 13) {
            printf("[!] second packet is not a uid!\n");
            exit(3);
        }
        *uidlen = buf[1];
        *uid = (uint8_t*) malloc(sizeof(uint8_t)*(*uidlen+1));
        *uidlen += 2;
        memcpy(*uid, buf, 2);
        read(fd, *uid+2, *uidlen);
        // terminate the cstring!
        *uid[*uidlen] = 0;
    }

}

int main() {

    int fd = -1;

    printf("[+] Reading secret key from vanity.sec\n");
    if ((fd = open("vanity.pub", O_RDONLY, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }

    int keylen, uidlen;
    uint8_t *key, *uid;
    readkey(fd, &key, &keylen, &uid, &uidlen);
    close(fd);

    printf("[*] Public Key Packet Size: %d\n", keylen);
    printf("[*] Uid Packet Size: %d\n", uidlen);
    printf("[*] Uid: %s\n", uid+2);

    uint8_t vanity[] = { 0xc0, 0x1a, 0xde };
    uint32_t timestamp = find_vanity(vanity, 3, key, keylen);
    if(!timestamp) {
        printf("[!] No key found in reasonable time range, giving up :(\n");
        return 1;
    }

    printf("[+] got it!\n");

    int i;
    printf("[*] timestamp: ");
    for(i = 0; i < 4; i++)
        printf("%02x", key[4+i]);
    printf("\n");

    // 20 bytes digest
    uint8_t digest[20];
    SHA1(key, keylen, digest);

    printf("[*] digest: ");
    for(i = 0; i < 20; i++)
        printf("%02x", digest[i]);
    printf("\n");

    printf("[+] Writing new public key to result.pub\n");
    if ((fd = open("result.pub", O_WRONLY|O_CREAT|O_TRUNC, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }
    write(fd, key, keylen);
    write(fd, uid, uidlen);
    close(fd);

    printf("[+] Reading secret key from vanity.sec\n");
    if ((fd = open("vanity.sec", O_RDONLY, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }

    readkey(fd, &key, &keylen, &uid, &uidlen);
    close(fd);

    printf("[*] Secret Key Packet Size: %d\n", keylen);
    printf("[*] Uid Packet Size: %d\n", uidlen);

    key[4] = (timestamp >> 24) & 0xff;
    key[5] = (timestamp >> 16) & 0xff;
    key[6] = (timestamp >> 8) & 0xff;
    key[7] = (timestamp) & 0xff;

    printf("[+] Writing new secret key to result.sec\n");
    if ((fd = open("result.sec", O_WRONLY|O_CREAT|O_TRUNC, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }
    write(fd, key, keylen);
    write(fd, uid, uidlen);
    close(fd);

    printf("[+] All done!\n");

    return 0;
}
