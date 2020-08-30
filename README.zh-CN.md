 

# PGP 靓号生成器

[English](README.md)

通过 GPG 枚举密钥进行指纹碰撞会消耗大量的时间。此生成器的原理，是先生成密钥，然后调整密钥的创建日期，来查找所想要的“靓号”后缀。

此生成器通过 CUDA 利用 GPU 实现密钥的高速运算。经过测试，在如下表的设备配置中，利用 GPU 实现的版本比利用 CPU 的版本快 3～4 倍。

|      CPU       |         GPU         |
| :------------: | :-----------------: |
| Intel i7-5500U | NVIDIA GeForce 940M |

## 版本

[Valodim](https://github.com/Valodim/pgp-vanity-keygen) 的原始版本已过时。GnuPG 2.1及更高版本不再使用 *secring* 进行密钥存储（参阅 [GnuPG](https://www.gnupg.org/faq/whats-new-in-2.1.html#nosecring)）。位于`traditional`分支的版本针对此更新进行了修改，并修改了输入逻辑。

位于`main`分支的版本需要安装 CUDA 才能编译。如果您不具备运行的条件，请利用通用的`traditional`分支版本。

您需要手动修改`Makefile`文件中的 `nvcc` 编译标志。这个标志是这样的：

`CUDAFLAGS= -O3 -rdc=true -gencode arch=compute_50,code=sm_50`

在如上表的设备配置中，显示卡`NVIDIA GeForce 940M`由于计算力过低，不与最新的 CUDA 默认配置兼容，因而加入了这些项。您可以首先去掉`-gencode`及其以后的所有内容并尝试运行。如果运行出现错误，请在[此处](https://developer.nvidia.com/cuda-gpus)查询您的显示卡计算力，并对应修改`-gencode`。

是否能够正常运行的一个简单的判断方法是，按照以下的使用方法编译后，输入`./pgpgen AA`。这条命令会在很短的时间内完成。检查命令的输出中后部分GPG的导入部分，观察密钥指纹是否以`AA`结尾。

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
