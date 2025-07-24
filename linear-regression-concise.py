from typing import Tuple
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = -4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(
    data_arrays: Tuple[torch.Tensor, ...],
    batch_size: int,
    is_train: bool = True
) -> data.DataLoader:
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# it = iter(data_iter)
# print(next(it))
# print(next(it))
# print(next(it))

# `nn` is short for neural network
from torch import nn

# 定义了一个神经网络模型。这里的 nn.Sequential 表示“顺序容器”，可以把多个层按顺序组合起来。
# nn.Linear(2, 1) 是一个全连接层，输入是 2 维，输出是 1 维。
# 所以 net 就是一个输入为 2 维、输出为 1 维的线性回归模型。
net = nn.Sequential(nn.Linear(2, 1))

#  w参数
net[0].weight.data.normal_(0, 0.01)
#  b参数
net[0].bias.data.fill_(0)

# Mean Squared Error Loss
loss = nn.MSELoss()

parameters = net.parameters()

# print([param for param in parameters])

# trainer 是优化器，用于更新模型参数。只关心参数，不关心模型结构
trainer = torch.optim.SGD(parameters, lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        y_hat = net(X) # 前向传播
        l = loss(y_hat, y) # 计算损失
        trainer.zero_grad() # 清空梯度
        l.backward() # 反向传播
        trainer.step() # 更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
b = net[0].bias.data

print(f'w: {w.reshape(true_w.shape)}')
print(f'b: {b}')
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)