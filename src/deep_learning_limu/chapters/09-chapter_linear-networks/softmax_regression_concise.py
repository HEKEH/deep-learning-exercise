import torch
from torch import nn
from deep_learning_limu.tools import load_data_fashion_mnist, train_ch3

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weight) # 前向传播

loss = nn.CrossEntropyLoss(reduction='none') # 与计算损失、反向转播相关

parameters = net.parameters()
trainer = torch.optim.SGD(parameters, lr=0.1) # 用于更新参数

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
