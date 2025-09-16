import torch
import matplotlib.pyplot as plt
import sys
import os
from torch import Tensor
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import load_data_fashion_mnist, train_ch3, predict_ch3

# 归一化指数函数
def softmax(X: Tensor) -> Tensor:
    X_exp = torch.exp(X) # n * m, n个样本，m个类别
    partition = X_exp.sum(1, keepdim=True) # n * 1
    return X_exp / partition # 分母应用广播机制，n * m

number_inputs = 28 * 28
number_outputs = 10
W = torch.normal(0, 0.01, size=(number_inputs, number_outputs), requires_grad=True)
b = torch.zeros(number_outputs, requires_grad=True)
# XW + b
def net(X: Tensor) -> Tensor:
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

# 作为loss函数
def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    return - torch.log(y_hat[range(len(y_hat)), y])

def sgd(params: List[Tensor], lr: float) -> None:
    with torch.no_grad():
        for param in params:
            grad = param.grad
            assert grad is not None
            param -= lr * grad
            grad.zero_()

lr = 0.1
# trainer
def updater() -> None:
    sgd([W, b], lr)


# 开始训练
if __name__ == "__main__":
    try:
        # 数据加载
        print("正在加载Fashion-MNIST数据集...")
        batch_size = 256
        train_iter, test_iter = load_data_fashion_mnist(batch_size)


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

