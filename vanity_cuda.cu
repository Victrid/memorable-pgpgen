#include "sha1.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

using namespace std;

__global__ void vanity_check(uint32_t timestamp_base, uint8_t* key, int keylen,
                             bool* resultblk, uint8_t* vanity, int vlen) {
    uint32_t timestamp = timestamp_base + 500 * blockIdx.x + threadIdx.x;
    uint8_t* sepkey;
    sepkey = (uint8_t*)malloc(keylen * sizeof(uint8_t));
    memcpy(sepkey, key, keylen * sizeof(uint8_t));
    sepkey[4] = (timestamp >> 24) & 0xff;
    sepkey[5] = (timestamp >> 16) & 0xff;
    sepkey[6] = (timestamp >> 8) & 0xff;
    sepkey[7] = (timestamp)&0xff;
    uint8_t digest[20];
    sha1ss(sepkey, keylen, digest);
    for (int i = 0; i < vlen; i++) {
        if (digest[19 - i] != vanity[vlen - 1 - i]) {
            break;
        }
        // if we're here, we found one! yay!
        if (i == vlen - 1) {
            resultblk[500 * blockIdx.x + threadIdx.x] = 1;
            return;
        }
    }
    resultblk[500 * blockIdx.x + threadIdx.x] = 0;
    return;
}

uint32_t find_vanity(uint8_t* vanity, int vlen, uint8_t* key, int keylen,
                     uint32_t limit) {

    time_t start = time(NULL);

    // 5 to 8 are the timestamp (the interesting bit!)
    uint32_t timestamp =
        (key[4] << 24) + (key[5] << 16) + (key[6] << 8) + key[7];

    uint8_t* GPUkey;
    cudaMallocManaged(&GPUkey, keylen * sizeof(uint8_t));
    memcpy(GPUkey, key, keylen * sizeof(uint8_t));

    uint8_t* GPUvanity;
    cudaMallocManaged(&GPUvanity, vlen * sizeof(uint8_t));
    memcpy(GPUvanity, vanity, vlen * sizeof(uint8_t));

    while (timestamp > limit) {
        // reduce by one
        timestamp -= 2000;
        key[4] = (timestamp >> 24) & 0xff;
        key[5] = (timestamp >> 16) & 0xff;
        key[6] = (timestamp >> 8) & 0xff;
        key[7] = (timestamp)&0xff;

        // calculate hash with new data

        bool* resultblk;
        cudaMallocManaged(&resultblk, sizeof(bool) * 2000);
        memset(resultblk, 0, 2000 * sizeof(bool));

        vanity_check<<<4, 500>>>(timestamp, GPUkey, keylen, resultblk,
                                 GPUvanity, vlen);

        cudaDeviceSynchronize();
        for (int i = 0; i < 2000; i++) {
            // if we're here, we found one! yay!
            if (resultblk[i]) {
                cudaFree(resultblk);
                return timestamp + i;
            }
        }
        cudaFree(resultblk);
    }
    cudaFree(GPUkey);
    cudaFree(GPUvanity);
    return 0;
}

void readkey(int fd, uint8_t** key, int* keylen) {

    uint8_t buf[3];
    {
        // make sure the first packet is a pubkey or seckey packet
        ssize_t rc = read(fd, buf, 3);
        if (rc < 0 || rc != 3) {
            perror("read");
            exit(-1);
        }

        if ((buf[0] & 0x3f) >> 2 != 6 && (buf[0] & 0x3f) >> 2 != 5) {
            printf("[!] packet is not a pubkey or seckey!\n");
            exit(2);
        }

        *keylen = (buf[1] << 8) + (buf[2]);
        // take into account the three bytes already read
        *keylen += 3;

        // assuming we work with 4096 keys at most
        *key = (uint8_t*)malloc(sizeof(uint8_t) * *keylen);
        // copy first three bytes from buf
        memcpy(*key, buf, 3);
        // read rest of the key
        rc = read(fd, *key + 3, (*keylen) - 3);
        if (rc < 0 || rc != (*keylen) - 3) {
            perror("read");
            exit(-1);
        }

        if ((*key)[3] != 0x04) {
            printf("[!] version number != 4\n");
            exit(2);
        }
    }
}

int main(int argc, char** argv) {

    int fd = -1;

    if (argc != 5) {
        printf("Usage: $0 keyring.pub keyring.sec timelimit vanitybytes\n");
        return 1;
    }

    printf("[+] Reading public key from %s\n", argv[1]);
    if ((fd = open(argv[1], O_RDONLY, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }

    int keylen;
    uint8_t* key;
    readkey(fd, &key, &keylen);
    close(fd);

    printf("[*] Public Key Packet Size: %d\n", keylen);

    uint8_t* vanity = (uint8_t*)argv[4];
    int vlen        = strlen(argv[4]);
    printf("[*] Searching for: 0x...");
    int i;
    for (i = 0; i < vlen; i++)
        printf("%02x", vanity[i]);
    printf("\n");

    uint32_t limit;
    {
        limit = strtol(argv[3], 0, 10);
        char buf[200];
        time_t ye      = limit;
        struct tm* tmp = localtime(&ye);
        strftime(buf, sizeof(buf), "%F", tmp);
        printf("[*] Searching down to %s\n", buf);
    }

    uint32_t timestamp = find_vanity(vanity, vlen, key, keylen, limit);
    if (!timestamp) {
        printf("[!] No key found in reasonable time range, giving up :(\n");
        return 1;
    }

    printf("[+] got it!\n");
    key[4] = (timestamp >> 24) & 0xff;
    key[5] = (timestamp >> 16) & 0xff;
    key[6] = (timestamp >> 8) & 0xff;
    key[7] = (timestamp)&0xff;

    printf("[*] timestamp: ");
    for (i = 0; i < 4; i++)
        printf("%02x", key[4 + i]);
    printf("\n");

    printf("[+] Writing new public key to result.pub\n");
    if ((fd = open("result.pub", O_WRONLY | O_CREAT | O_TRUNC,
                   S_IRUSR | S_IWUSR)) == -1) {
        printf("[-] open() failed");
        return 2;
    }
    ssize_t rc = write(fd, key, keylen);
    if (rc < 0 || rc != keylen) {
        perror("write");
        exit(-1);
    }
    rc = write(fd,
               "\xb4\x22"
               "fake uid, replace with a valid one",
               36);
    if (rc < 0 || rc != 36) {
        perror("write");
        exit(-1);
    }
    close(fd);

    printf("[+] Reading secret key from %s\n", argv[2]);
    if ((fd = open(argv[2], O_RDONLY, 0)) == -1) {
        printf("[-] open() failed");
        return 2;
    }

    readkey(fd, &key, &keylen);
    close(fd);

    printf("[*] Secret Key Packet Size: %d\n", keylen);

    key[4] = (timestamp >> 24) & 0xff;
    key[5] = (timestamp >> 16) & 0xff;
    key[6] = (timestamp >> 8) & 0xff;
    key[7] = (timestamp)&0xff;

    printf("[+] Writing new secret key to result.sec\n");
    if ((fd = open("result.sec", O_WRONLY | O_CREAT | O_TRUNC,
                   S_IRUSR | S_IWUSR)) == -1) {
        printf("[-] open() failed");
        return 2;
    }
    rc = write(fd, key, keylen);
    if (rc < 0 || rc != keylen) {
        perror("write");
        exit(-1);
    }

    rc = write(fd,
               "\xb4\x22"
               "fake uid, replace with a valid one",
               36);
    if (rc < 0 || rc != 36) {
        perror("write");
        exit(-1);
    }
    close(fd);

    printf("[+] All done!\n");

    return 0;
}
