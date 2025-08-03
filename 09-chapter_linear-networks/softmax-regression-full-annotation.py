"""
Softmax回归 - 从零开始实现
这是一个完整的机器学习项目，教你如何从零开始构建一个图像分类模型

学习目标：
1. 理解机器学习的完整流程
2. 学会如何加载和处理数据
3. 理解模型构建和训练过程
4. 学会如何评估和改进模型

作者：为初学者优化
"""

# ==================== 第一步：导入必要的库 ====================
# 这些库就像工具箱，每个都有特定的用途
import torch  # PyTorch：深度学习框架，就像我们的"大脑"
import matplotlib  # 绘图库，用来可视化结果
import matplotlib.pyplot as plt  # 绘图工具
import sys  # 系统相关功能
import os  # 文件和目录操作
from torch import Tensor  # PyTorch的张量类型
from d2l import torch as d2l  # 深度学习工具库
from typing import List, Tuple, Optional, Union, Any  # 类型提示
import torchvision  # 计算机视觉库
from torchvision import transforms  # 图像变换工具
from torch.utils.data import DataLoader  # 数据加载器

# ==================== 第二步：配置显示环境 ====================
# 这一步确保图形能够正确显示，就像设置显示器一样
def setup_matplotlib() -> bool:
    """
    配置matplotlib以确保图形能正常显示
    就像设置电视机的显示模式一样
    """
    try:
        # 检查是否在Jupyter环境中（就像检查是否在网页中运行）
        if 'ipykernel' in sys.modules:
            matplotlib.use('TkAgg')  # 使用TkAgg后端
            # 在Jupyter中设置内联显示（图形直接显示在网页中）
            try:
                from IPython import get_ipython
                get_ipython().run_line_magic('matplotlib', 'inline')
            except (NameError, ImportError):
                print("IPython不可用，跳过内联显示设置")
        else:
            # 在普通Python环境中使用TkAgg（弹出窗口显示）
            matplotlib.use('TkAgg')

        # 设置图形参数（就像设置画布的大小和样式）
        plt.rcParams['figure.figsize'] = (10, 6)  # 图形大小
        plt.rcParams['font.size'] = 12  # 字体大小
        plt.rcParams['axes.grid'] = True  # 显示网格
        plt.rcParams['grid.alpha'] = 0.3  # 网格透明度

        print("✅ matplotlib配置完成 - 图形可以正常显示了！")
        return True
    except Exception as e:
        print(f"❌ matplotlib配置失败: {e}")
        return False

# 初始化matplotlib
print("🔧 正在配置显示环境...")
setup_matplotlib()

# ==================== 第三步：定义辅助工具类 ====================

class Accumulator:
    """
    累加器类 - 用来统计训练过程中的各种指标
    就像记账本一样，记录所有的数据
    """
    def __init__(self, n: int) -> None:
        """
        初始化累加器
        n: 需要统计的指标数量（比如损失、准确率等）
        """
        self.data = [0.0] * n  # 创建n个计数器，初始值为0

    def add(self, *args: float) -> None:
        """
        添加新的数据
        *args: 要添加的数值（比如当前批次的损失、准确率等）
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        """重置所有计数器为0"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        """获取第idx个指标的值"""
        return self.data[idx]

class Animator:
    """
    动画器类 - 实时显示训练过程
    就像实时监控器，显示训练进度
    """
    def __init__(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 legend: Optional[List[str]] = None, xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None, xscale: str = 'linear',
                 yscale: str = 'linear', fmts: Tuple[str, ...] = ('-', 'm--', 'g-.', 'r:'),
                 nrows: int = 1, ncols: int = 1, figsize: Tuple[float, float] = (10, 6)) -> None:
        """
        初始化动画器
        xlabel: x轴标签（比如"训练轮数"）
        ylabel: y轴标签（比如"准确率"）
        legend: 图例（比如["训练损失", "训练准确率", "测试准确率"]）
        """
        # 增量地绘制多条线
        if legend is None:
            legend = []

        # 尝试使用SVG显示，如果失败则使用普通显示
        try:
            d2l.use_svg_display()
        except:
            print("SVG显示不可用，使用普通显示")

        # 创建图形和坐标轴
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # 使用lambda函数捕获参数，用于配置坐标轴
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        # 检查display模块是否可用（用于Jupyter环境）
        try:
            from IPython import display
            self.display_available = True
            self.display = display
        except ImportError:
            self.display_available = False
            print("IPython display不可用，将使用plt.show()")

    def add(self, x: Union[float, List[float]], y: Union[float, List[float]]) -> None:
        """
        添加新的数据点并更新图形
        x: x轴数据（比如当前轮数）
        y: y轴数据（比如当前的损失和准确率）
        """
        # 处理数据格式
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        # 添加新数据点
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

# ==================== 第四步：加载数据 ====================
# 这一步就像准备学习材料，数据是机器学习的"食物"

print("📊 正在加载Fashion-MNIST数据集...")
print("💡 提示：Fashion-MNIST是一个包含10种衣服类型的图像数据集")
print("   包括：T恤、裤子、连衣裙、外套、凉鞋、衬衫、运动鞋、包、靴子、凉鞋")

batch_size = 256  # 批次大小：每次处理256张图片

def load_data_fashion_mnist(batch_size: int, resize: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    加载Fashion-MNIST数据集
    就像从图书馆借书一样，我们需要把数据"借"到电脑里

    参数：
    batch_size: 批次大小（每次处理多少张图片）
    resize: 是否调整图片大小

    返回：
    train_iter: 训练数据加载器
    test_iter: 测试数据加载器
    """
    # 定义数据变换（就像给图片做预处理）
    trans = [transforms.ToTensor()]  # 将图片转换为张量
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 调整图片大小
    trans = transforms.Compose(trans)  # 组合所有变换

    # 下载并加载训练数据
    print("📥 下载训练数据...")
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)

    # 下载并加载测试数据
    print("📥 下载测试数据...")
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    # 创建数据加载器（就像把书整理成小册子）
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

    print(f"✅ 数据加载完成！")
    print(f"   训练集：{len(mnist_train)} 张图片")
    print(f"   测试集：{len(mnist_test)} 张图片")
    print(f"   批次大小：{batch_size}")

    return train_iter, test_iter

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# ==================== 第五步：定义模型 ====================
# 这一步就像设计一个"大脑"，告诉电脑如何思考

print("\n🧠 正在构建模型...")

# 定义模型参数
number_inputs = 28 * 28  # 输入维度：每张图片是28x28像素，展平后是784
number_outputs = 10      # 输出维度：10种衣服类型

print(f"📐 模型参数：")
print(f"   输入维度：{number_inputs} (28×28像素)")
print(f"   输出维度：{number_outputs} (10种衣服类型)")

# 初始化权重和偏置（就像给大脑设置初始的"思考方式"）
W = torch.normal(0, 0.01, size=(number_inputs, number_outputs), requires_grad=True)  # 权重矩阵
b = torch.zeros(number_outputs, requires_grad=True)  # 偏置向量

print(f"   权重矩阵形状：{W.shape}")
print(f"   偏置向量形状：{b.shape}")

# 定义Softmax函数（就像投票选举，把分数转换成概率）
def softmax(X: Tensor) -> Tensor:
    """
    Softmax函数：将任意实数转换为概率分布
    就像把票数转换成当选概率一样

    参数：
    X: 输入张量（原始分数）

    返回：
    概率分布（所有值都是正数，且和为1）
    """
    X_exp = torch.exp(X)  # 计算指数
    partition = X_exp.sum(1, keepdim=True)  # 计算分母（归一化因子）
    return X_exp / partition  # 返回概率分布

# 定义网络模型（就像定义大脑的思考过程）
def net(X: Tensor) -> Tensor:
    """
    网络模型：完整的预测过程
    就像大脑的思考过程：接收输入 → 处理 → 输出结果

    参数：
    X: 输入图片（形状：[批次大小, 784]）

    返回：
    预测结果（形状：[批次大小, 10]）
    """
    # 1. 线性变换：X * W + b
    linear_output = torch.matmul(X.reshape(-1, W.shape[0]), W) + b
    # 2. Softmax激活：将分数转换为概率
    return softmax(linear_output)

print("✅ 模型构建完成！")

# ==================== 第六步：定义损失函数和评估指标 ====================

# 定义交叉熵损失函数（就像计算预测错误的程度）
def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    """
    交叉熵损失函数：衡量预测与真实值的差距
    就像考试评分，错误越多分数越低

    参数：
    y_hat: 预测结果（概率分布）
    y: 真实标签

    返回：
    损失值（越小越好）
    """
    return -torch.log(y_hat[range(len(y_hat)), y])

# 定义准确率计算函数（就像计算考试的正确率）
def accuracy(y_hat: Tensor, y: Tensor) -> float:
    """
    计算准确率：预测正确的比例
    就像计算考试的正确率

    参数：
    y_hat: 预测结果
    y: 真实标签

    返回：
    准确率（0-1之间，越大越好）
    """
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(dim=1)  # 取概率最大的类别作为预测结果
    cmp = y_hat.type(y.dtype) == y  # 比较预测和真实值
    return float(cmp.type(y.dtype).sum())  # 返回正确预测的数量

# ==================== 第七步：定义训练和评估函数 ====================

def evaluate_accuracy(net: Union[torch.nn.Module, Any], data_iter: DataLoader) -> float:
    """
    评估模型在指定数据集上的准确率
    就像给学生做测试，看看学得怎么样

    参数：
    net: 网络模型
    data_iter: 数据迭代器

    返回：
    准确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式

    metric = Accumulator(2)  # 创建累加器，记录正确数量和总数量

    with torch.no_grad():  # 不计算梯度（节省内存）
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # 累加正确数量和总数量

    return metric[0] / metric[1]  # 返回准确率

def train_epoch_ch3(net: Union[torch.nn.Module, Any], train_iter: DataLoader,
                   loss: Any, updater: Union[torch.optim.Optimizer, Any]) -> Tuple[float, float]:
    """
    训练一个完整的epoch（遍历所有训练数据一次）
    就像学生做一次完整的练习

    参数：
    net: 网络模型
    train_iter: 训练数据迭代器
    loss: 损失函数
    updater: 优化器

    返回：
    (平均损失, 平均准确率)
    """
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式

    metric = Accumulator(3)  # 记录损失、正确数量、总数量

    for X, y in train_iter:  # 遍历每个批次
        y_hat = net(X)  # 前向传播：计算预测结果
        l = loss(y_hat, y)  # 计算损失

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch优化器
            updater.zero_grad()  # 清空梯度
            l.mean().backward()  # 反向传播：计算梯度
            updater.step()  # 更新参数
        else:
            # 使用自定义优化器
            l.mean().backward()  # 反向传播
            updater()  # 更新参数

        # 记录统计信息
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率

# ==================== 第八步：定义优化器 ====================

def sgd(params: List[Tensor], lr: float) -> None:
    """
    随机梯度下降优化器
    就像学生根据错误调整学习方法

    参数：
    params: 需要更新的参数列表
    lr: 学习率（学习步长）
    """
    with torch.no_grad():  # 不计算梯度
        for param in params:
            grad = param.grad  # 获取梯度
            assert grad is not None
            param -= lr * grad  # 更新参数：新参数 = 旧参数 - 学习率 × 梯度
            grad.zero_()  # 清空梯度

# 设置学习率
lr = 0.1  # 学习率：控制每次更新的步长

def updater() -> None:
    """更新函数：调用SGD优化器更新模型参数"""
    sgd([W, b], lr)

print(f"🔧 优化器设置完成！学习率：{lr}")

# ==================== 第九步：主训练函数 ====================

def train_ch3(net: Union[torch.nn.Module, Any], train_iter: DataLoader,
              test_iter: DataLoader, loss: Any, num_epochs: int,
              updater: Union[torch.optim.Optimizer, Any]) -> None:
    """
    主训练函数：完整的训练过程
    就像学生的完整学习过程

    参数：
    net: 网络模型
    train_iter: 训练数据
    test_iter: 测试数据
    loss: 损失函数
    num_epochs: 训练轮数
    updater: 优化器
    """
    print(f"\n🚀 开始训练！总共 {num_epochs} 轮")
    print("📈 训练过程将实时显示损失和准确率变化...")

    # 创建动画器，用于实时显示训练进度
    animator = Animator(
        xlabel="训练轮数",
        ylabel="指标值",
        xlim=[1, num_epochs],
        ylim=[0.0, 1.0],
        legend=['训练损失', '训练准确率', '测试准确率']
    )

    for epoch in range(num_epochs):
        print(f"\n🔄 第 {epoch + 1} 轮训练...")

        # 训练一个epoch
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)

        # 评估测试集
        test_acc = evaluate_accuracy(net, test_iter)

        # 显示当前轮数的结果
        print(f"   训练损失：{train_loss:.4f}")
        print(f"   训练准确率：{train_acc:.4f}")
        print(f"   测试准确率：{test_acc:.4f}")

        # 更新动画器
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))

    # 验证最终结果
    print(f"\n✅ 训练完成！最终结果：")
    print(f"   训练损失：{train_loss:.4f}")
    print(f"   训练准确率：{train_acc:.4f}")
    print(f"   测试准确率：{test_acc:.4f}")

    # 检查训练是否成功
    assert train_loss < 0.5, f"训练损失过高: {train_loss}"
    assert train_acc <= 1 and train_acc > 0.7, f"训练准确率异常: {train_acc}"
    assert test_acc <= 1 and test_acc > 0.7, f"测试准确率异常: {test_acc}"
    print("🎉 所有指标都在合理范围内，训练成功！")

# ==================== 第十步：预测和可视化函数 ====================

def predict_ch3(net: Union[torch.nn.Module, Any], test_iter: DataLoader, n: int = 6) -> None:
    """
    预测函数：展示模型的预测结果
    就像让学生展示学习成果

    参数：
    net: 训练好的网络模型
    test_iter: 测试数据
    n: 要显示的图片数量
    """
    print(f"\n🔍 展示预测结果（显示 {n} 张图片）...")

    # 获取一批测试数据
    for X, y in test_iter:
        break

    # 获取真实标签和预测标签
    trues = d2l.get_fashion_mnist_labels(y)  # 真实标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))  # 预测标签

    # 创建标题（显示真实值和预测值）
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    # 计算显示布局
    rows = (n + 5) // 6  # 行数
    cols = min(n, 6)     # 列数（最多6列）

    # 显示图片和预测结果
    d2l.show_images(X[0:n].reshape(n, 28, 28), rows, cols, titles=titles)

    print("✅ 预测结果展示完成！")
    print("💡 提示：每张图片上方显示真实标签，下方显示预测标签")

# ==================== 第十一步：主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Softmax回归 - 从零开始实现")
    print("📚 这是一个完整的机器学习项目")
    print("=" * 60)

    try:
        # 设置训练参数
        num_epochs = 6  # 训练轮数
        print(f"\n⚙️ 训练参数：")
        print(f"   训练轮数：{num_epochs}")
        print(f"   批次大小：{batch_size}")
        print(f"   学习率：{lr}")

        # 开始训练
        print(f"\n🎯 开始训练模型...")
        train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

        # 展示预测结果
        print(f"\n🎨 展示模型预测能力...")
        predict_ch3(net, test_iter, 18)  # 显示18张图片的预测结果

        # 训练完成
        print(f"\n🎉 项目完成！")
        print(f"📊 训练过程图形已保存到 ./plots/training_progress.png")

        # 保持图形显示（在非Jupyter环境中）
        if not 'ipykernel' in sys.modules:
            plt.show(block=True)
            input("按回车键关闭图形窗口...")

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示：如果遇到错误，请检查：")
        print("   1. 网络连接是否正常（需要下载数据集）")
        print("   2. 是否有足够的磁盘空间")
        print("   3. Python环境是否正确配置")

print("\n" + "=" * 60)
print("📖 学习总结：")
print("   1. 数据加载：准备学习材料")
print("   2. 模型构建：设计思考方式")
print("   3. 损失函数：定义错误标准")
print("   4. 优化器：改进学习方法")
print("   5. 训练过程：反复练习改进")
print("   6. 评估结果：检查学习效果")
print("=" * 60)
