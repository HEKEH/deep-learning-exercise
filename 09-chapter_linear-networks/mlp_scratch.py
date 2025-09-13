# MLP: multi-layer perceptron
import torch
from torch import nn
from softmax_regression import load_data_fashion_mnist, train_ch3

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, nums_outputs, num_hiddens = 784, 10, 256
# n * num_inputs -> n * num_hiddens -> n * num_outputs
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) # randn: random normal
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, nums_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(nums_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # 这里“@”代表矩阵乘法
    return H @ W2 + b2

loss = nn.CrossEntropyLoss(reduction='none') # 与计算损失、反向转播相关

trainer = torch.optim.SGD(params, lr=0.1) # 用于更新参数

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
