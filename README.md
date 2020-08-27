# Memorable PGP key ID Generator

[简体中文](README.zh-CN.md)

This works by generating a key and then adjusting the creation date until it finds something that hashes to the desired ending values.

## Old version

The original version by [Valodim](https://github.com/Valodim/pgp-vanity-keygen) is outdated. GnuPG's 2.1 and newer version don't use the *secring* for secret keys storage anymore (*Cf.* [GnuPG](https://www.gnupg.org/faq/whats-new-in-2.1.html#nosecring)). The generation script is modified to fit the update. The input method is also modified for easy use.

## Usage

Build with

```sh
make
```

Generate your key with a preferred suffix (e.g. `0xDEADBEEF`). The length of suffix should be even.

```sh
./pgpgen DEADBEEF
```

You may run this for multiple times, each of the result will be added to the `private.key` file.

All these keys will be in an odd state since there is no valid UID and no signatures. Import them directly may ruin your keychain, or just impossible due to GnuPG rules.

To protect your existing keyring, Import the result into a temporary gpg homedir (e.g. `result`):

```sh
mkdir --mode 700 result
gpg --homedir ./result --allow-non-selfsigned-uid --import private.key
```

View your generated keys:

```sh
gpg --homedir ./result --list-keys
```

To modify this into a *common* PGP key, edit the key (e.g. `5E1EC210DEADBEEF`):

```sh
gpg --homedir ./result --edit-key 5E1EC210DEADBEEF
```

Enter these commands to modify as you wish:

```sh
adduid (interactive: user description)

uid 1
deluid (select the fake uid and delete it)

addkey (create an RSA signing key)
addkey (create an RSA encryption key)
passwd (change your password)
save   (save it and quit the edit)
```

Export the *common* key (e.g. `5E1EC210DEADBEEF`) to file (e.g. `good.key`) with:

```sh
gpg --homedir ./result --armor --output good.key --export-secret-keys 5E1EC210DEADBEEF
```

then import this key file (e.g. `good.key`) as a normal one to your own keychain.

```sh
gpg --import good.key
```

To clean the directory to avoid unwanted leakage, run

```sh
make clean
```

to clean the generated directory. 

You also need to remove the `result` folder, and `good.key` file yourself.
