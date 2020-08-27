 

# PGP 靓号生成器

[English](README.md)

通过 GPG 枚举密钥进行指纹碰撞会消耗大量的时间。此生成器的原理，是先生成密钥，然后调整密钥的创建日期，来查找所想要的“靓号”后缀。

## 旧版本

[Valodim](https://github.com/Valodim/pgp-vanity-keygen) 的原始版本已过时。GnuPG 2.1及更高版本不再使用 *secring* 进行密钥存储（参阅 [GnuPG](https://www.gnupg.org/faq/whats-new-in-2.1.html#nosecring)）。此版本针对此更新进行了修改，并修改了输入逻辑。

## 使用方法

编译：

```sh
make
```

生成带有特定后缀的密钥（例如，“ 0xDEADBEEF”）。注意，此后缀必须为偶数位的。

```sh
./pgpgen DEADBEEF
```

您可以多次运行此命令，每次运行的结果将被添加到`private.key`文件中。

由于没有有效的 UID 和签名，所有这些密钥由于 GnuPG 规则都不可被导入。强行导入它们可能会破坏您的钥匙串。

为了保护您现有的钥匙串，请将结果导入到临时的 gpg 目录中（例如此处 的 `result`）：

```sh
mkdir --mode 700 result
gpg --homedir ./result --allow-non-selfsigned-uid --import private.key
```

查看您生成的密钥：

```sh
mkdir --mode 700 result
gpg --homedir ./result --list-keys
```

要将其修改为*通用* PGP 密钥，请编辑密钥（例如 `5E1EC210DEADBEEF`）：

```sh
gpg --homedir ./result --edit-key 5E1EC210DEADBEEF
```

输入以下命令进行修改：

```sh
adduid（交互式：您的用户名与邮箱）

uid 1
deluid（选择预生成 uid 并将其删除）

addkey（创建 RSA 签名密钥）
addkey（创建 RSA 加密密钥）
passwd（更改密码）
save  （保存并退出编辑）
```

使用以下命令将*通用* PGP 密钥（例如 `5E1EC210DEADBEEF`）导出到文件（例如 `good.key`）中：

```sh
gpg --homedir ./result --armor --output good.key --export-secret-keys 5E1EC210DEADBEEF
```

然后将此密钥文件（例如 `good.key`）以常规方法导入到您自己的钥匙串中。

```sh
gpg --import good.key
```

若要清理目录以避免不必要的泄漏，请运行

```sh
make clean
```

您还需要自己删除`result`文件夹和`good.key`文件。
