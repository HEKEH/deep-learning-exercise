import torch
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from torch import Tensor
from d2l import torch as d2l
from typing import List
import torchvision
from torchvision import transforms

# 配置matplotlib后端
def setup_matplotlib():
    """配置matplotlib以确保图形能正常显示"""
    try:
        # 检查是否在Jupyter环境中
        if 'ipykernel' in sys.modules:
            matplotlib.use('TkAgg')
            # 在Jupyter中设置内联显示
            try:
                from IPython import get_ipython
                get_ipython().run_line_magic('matplotlib', 'inline')
            except (NameError, ImportError):
                print("IPython不可用，跳过内联显示设置")
        else:
            # 在普通Python环境中使用TkAgg
            matplotlib.use('TkAgg')

        # 设置图形参数
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        print("matplotlib配置完成")
        return True
    except Exception as e:
        print(f"matplotlib配置失败: {e}")
        return False

# 初始化matplotlib
setup_matplotlib()

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(10, 6)):
        # 增量地绘制多条线
        if legend is None:
            legend = []

        # 尝试使用SVG显示，如果失败则使用普通显示
        try:
            d2l.use_svg_display()
        except:
            print("SVG显示不可用，使用普通显示")

        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        # 检查display模块是否可用
        try:
            from IPython import display
            self.display_available = True
            self.display = display
        except ImportError:
            self.display_available = False
            print("IPython display不可用，将使用plt.show()")

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 清除当前图形并重新绘制
        self.axes[0].cla()
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt, linewidth=2)

        self.config_axes()

        # 根据环境选择显示方式
        if self.display_available:
            try:
                self.display.display(self.fig)
                self.display.clear_output(wait=True)
            except:
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.show(block=False)
            plt.pause(0.1)

        # 保存图形到文件（可选）
        try:
            os.makedirs('./plots', exist_ok=True)
            plt.savefig('./plots/training_progress.png', dpi=300, bbox_inches='tight')
        except:
            pass

# 数据加载
print("正在加载Fashion-MNIST数据集...")
batch_size = 256
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=4))
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 以下是训练的主体内容

number_inputs = 28 * 28
number_outputs = 10
W = torch.normal(0, 0.01, size=(number_inputs, number_outputs), requires_grad=True)
b = torch.zeros(number_outputs, requires_grad=True)
# XW + b

# 归一化指数函数
def softmax(X: Tensor) -> Tensor:
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 应用广播机制

def net(X: Tensor):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

# 作为loss函数
def cross_entropy(y_hat: Tensor, y: Tensor) -> float:
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat: Tensor, y: Tensor) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算net在指定测试数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.mean().backward()
            updater()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2] # loss, accuracy

def sgd(params: List[Tensor], lr: float):
    with torch.no_grad():
        for param in params:
            grad = param.grad
            assert grad is not None
            param -= lr * grad
            grad.zero_()

lr = 0.1
def updater():
    sgd([W, b], lr)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0.0, 1.0], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    rows = (n + 5) // 6
    cols = min(n, 6)
    d2l.show_images(X[0:n].reshape(n, 28, 28), rows, cols, titles=titles)


# 开始训练
if __name__ == "__main__":
    try:
        num_epochs = 6
        train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

        predict_ch3(net, test_iter, 18)

        # 训练完成后保持图形显示
        print("\n训练完成！图形已保存到 ./plots/training_progress.png")
        if not 'ipykernel' in sys.modules:
            plt.show(block=True)
            input("按回车键关闭图形窗口...")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
