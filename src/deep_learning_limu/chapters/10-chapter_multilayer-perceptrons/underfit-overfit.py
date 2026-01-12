import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from deep_learning_limu.tools import Accumulator, Animator, train_epoch_ch3

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
# 实际上起作用的只有前4维
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 200 * 1 的随机数
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)

# np.arange(max_degree).reshape(1, -1) 得到 1 * 20。poly_features为 200 * 20 的矩阵。每一行是 1, x, x^2, ..., x^19 的值。
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n) = (n - 1)!

# labels是 (200,) 的向量
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        # print(X, y, out, l)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# train_features的size为 (100, 20)，test_features的size为 (100, 20), train_labels的size为 100, test_labels的size为 100
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    # Mean Squared Error 均方误差。所有样本的损失的平均值
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    # train_labels.reshape(-1, 1)得到 (100, 1) 的矩阵
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# # 正常,从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
# # 欠拟合
# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
# 过拟合
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1200)


