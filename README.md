Generate vanity PGP key IDs
===

This works by generating a key and then adjusting the creation date
until it finds something that hashes to the desired ending values.

The `vanitywrapper` script sort of works, although I found it easiest to
do some of them by hand:

* Generate a valid public/private key pair
* Run `vanity` by hand and adjust the inputs to find something interesting

If `vanity` exits successfully it will write a `result.sec` and `result.pub`
file.  These will be in an odd state since there is no valid UID and
no signatures.

To protect your existing keyring, create a new `gpg` directory and
work on the keys there.

```
mkdir --mode 700 gpg
gpg --homedir ./gpg --allow-non-selfsigned-uid --import result.pub result.sec
gpg --homedir ./gpg --edit-key yourkeyid
adduid
(answer the questions)
uid 1
deluid
addkey
(create an RSA signing key, this will take a while)
addkey
(create an RSA encryption key, this will take a while)
trust
5 (ultimate)
passwd
(input your password)
save
```

