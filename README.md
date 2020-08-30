# Memorable PGP key ID Generator

[简体中文](README.zh-CN.md)

This works by generating a key and then adjusting the creation date until it finds something that hashes to the desired ending values.

This generator uses GPU to achieve high-speed operation of keys through CUDA. After testing, in the device configuration shown in the table below, the version implemented with GPU is 3 to 4 times faster than the version implemented with CPU.

|      CPU       |         GPU         |
| :------------: | :-----------------: |
| Intel i7-5500U | NVIDIA GeForce 940M |

## Versions

The original version by [Valodim](https://github.com/Valodim/pgp-vanity-keygen) is outdated. GnuPG's 2.1 and newer version don't use the *secring* for secret keys storage anymore (*Cf.* [GnuPG](https://www.gnupg.org/faq/whats-new-in-2.1.html#nosecring)). The version located in the `traditional` branch has been modified for this update and the input logic has been modified for easy use.

The version located in the `main` branch requires CUDA to be installed to compile. If you do not have the conditions to run, please use the general `traditional` branch version.

You need to manually modify the `nvcc` compilation flag in the `Makefile` file. The flag is like this:

`CUDAFLAGS= -O3 -rdc=true -gencode arch=compute_50,code=sm_50`

In the device configuration in the above table, the graphics card `NVIDIA GeForce 940M` is not compatible with the latest CUDA default configuration due to low compute capability, so these items have been added to compile. You should first remove `-gencode` and everything afterwards and try to run it. If error occured, please check the compute capability of your graphics card via [here](https://developer.nvidia.com/cuda-gpus), and modify `-gencode` accordingly.

A simple way to check whether it can run normally is to compile according to the following usage method and enter `./pgpgen AA`. This command will be completed in a short time. Check the import message of the GPG at the latter part in the output of the command and check whether the key fingerprint ends with `AA`.

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
