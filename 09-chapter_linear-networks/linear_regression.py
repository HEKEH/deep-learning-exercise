import random
import torch
from torch import Tensor
from typing import Generator, List, Tuple

def synthetic_data(
    w: Tensor,
    b: float,
    num_examples: int
) -> Tuple[Tensor, Tensor]:
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size: int, features: Tensor, labels: Tensor) -> Generator[Tuple[Tensor, Tensor], None, None]:
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]

# 从输入层，到输出层，前向传播
def linreg(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params: List[Tensor], lr: float) -> None:
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            grad = param.grad
            assert grad is not None
            param -= lr * grad
            grad.zero_()

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

batch_size = 10

lr = 0.03 # learning rate
num_epochs = 3
net = linreg # 神经网络，从输入层到输出层
loss = squared_loss # 损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b) # 前向传播
        l = loss(y_hat, y) # 计算损失
        l.mean().backward() # 反向传播
        sgd([w, b], lr) # 更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w: {w.reshape(true_w.shape)}')
print(f'b: {b}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
