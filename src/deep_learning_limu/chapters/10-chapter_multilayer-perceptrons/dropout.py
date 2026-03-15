# 丢弃法
# 是一种正则化方法，用于防止过拟合
# 它通过随机丢弃一些神经元来防止过拟合

import torch
from torch import nn

from deep_learning_limu.tools import load_data_fashion_mnist, train_ch3


def dropout_layer(X: torch.Tensor, dropout: float) -> torch.Tensor:
    assert 0 <= dropout <= 1, "dropout must be a float in the range [0, 1]"
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1 - dropout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hiddens1,
        num_hiddens2,
        dropout1,
        dropout2,
        is_training=True,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(
            (-1, self.num_inputs)
        )  # 将X的形状从(batch_size, 1, 28, 28) 变为 (batch_size, 784)
        H1 = self.relu(self.lin1(X))
        if self.training == True:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction="none")  # 交叉熵计算
train_iter, test_iter = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
