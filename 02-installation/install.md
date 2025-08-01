我们需要配置一个环境来运行 Python、Jupyter Notebook、相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

## 安装 Miniconda

最简单的方法就是安装依赖Python 3.x的[Miniconda](https://conda.io/en/latest/miniconda.html)。
如果已安装conda，则可以跳过以下步骤。访问Miniconda网站，根据Python3.x版本确定适合的版本。

如果我们使用macOS，假设Python版本是3.9（我们的测试版本），将下载名称包含字符串“MacOSX”的bash脚本，并执行以下操作：

```bash
# 以Intel处理器为例，文件名可能会更改
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```

如果我们使用Linux，假设Python版本是3.9（我们的测试版本），将下载名称包含字符串“Linux”的bash脚本，并执行以下操作：

```bash
# 文件名可能会更改
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```

接下来，初始化终端Shell，以便我们可以直接运行`conda`。

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开当前的shell。并使用下面的命令创建一个新的环境：

```bash
conda create --name d2l python=3.9 -y
```

现在激活 `d2l` 环境：

```bash
conda activate d2l
```

## 安装深度学习框架和`d2l`软件包

在安装深度学习框架之前，请先检查计算机上是否有可用的GPU。
例如可以查看计算机是否装有NVIDIA GPU并已安装[CUDA](https://developer.nvidia.com/cuda-downloads)。
如果机器没有任何GPU，没有必要担心，因为CPU在前几章完全够用。
但是，如果想流畅地学习全部章节，请提早获取GPU并且安装深度学习框架的GPU版本。

我们可以按如下方式安装PyTorch的CPU或GPU版本：

```bash
pip install torch==1.12.0
pip install torchvision==0.13.0
```
