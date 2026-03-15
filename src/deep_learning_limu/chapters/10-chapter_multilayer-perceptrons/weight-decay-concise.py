import torch
from deep_learning_limu.tools import (
    Animator,
    evaluate_loss,
    load_array,
    synthetic_data,
)

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)

# print("true_w shape:", true_w.shape)
# print("true_b:", true_b)
# print("train_data shape:", train_data[0].shape, train_data[1].shape)
# print("test_data shape:", test_data[0].shape, test_data[1].shape)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w: torch.Tensor):
    return torch.sum(w.pow(2)) / 2


def train_concise(wd: float):
    net = torch.nn.Sequential(torch.nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
        # print(param.data.shape)
    loss = torch.nn.MSELoss(reduction='mean')
    num_epochs = 100
    lr = 0.003
    trainer = torch.optim.SGD(
        [{"params": net[0].weight, "weight_decay": wd}, {"params": net[0].bias}], lr=lr
    )
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', net[0].weight.norm().item())

# train_concise(wd=0)
train_concise(wd=3)
