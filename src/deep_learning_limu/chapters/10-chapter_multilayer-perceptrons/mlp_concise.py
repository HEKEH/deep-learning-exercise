import torch
from torch import nn
from deep_learning_limu.tools import load_data_fashion_mnist, train_ch3

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, nums_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, nums_outputs))

def init_weights(m):
    if type(m) == nn.Linear: # 上面的 nn.Linear(num_inputs, num_hiddens) 之类操作时触发
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none') # 交叉熵计算。与计算损失、反向转播相关
trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 用于更新参数

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
