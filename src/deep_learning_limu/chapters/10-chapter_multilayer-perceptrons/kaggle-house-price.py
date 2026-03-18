# 房价预测
# 学习点:
# 1. 使用K折交叉验证来训练
# 2. 使用Adam优化器
# 3. 使用 (x - x.mean()) / (x.std()) 标准化数据
# 4. 使用 pd.get_dummies(all_features, dummy_na=True) 处理离散特征
# 5. k折交叉验证训练数据的目的是调整超参数

import pandas as pd
import torch
from torch import nn
from typing import Optional, Tuple, List

from deep_learning_limu.tools import load_array, plot

DATA_DIR = "data/kaggle-house-price-data"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SUBMISSION_CSV = f"{DATA_DIR}/submission.csv"

PLOT_FILE = "kaggle_house_price_rmse.png"

# 默认超参数（可先用 K 折验证再调）
DEFAULT_K = 5
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 5
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_BATCH_SIZE = 64

LOSS_FN = nn.MSELoss()


def load_and_preprocess(
    train_csv: str = TRAIN_CSV,
    test_csv: str = TEST_CSV,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """读取 CSV 并完成特征工程，返回张量形式的训练/测试特征与训练标签。"""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 去掉 Id；训练集还要去掉 SalePrice
    all_features = pd.concat((train_df.iloc[:, 1:-1], test_df.iloc[:, 1:]))

    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    # 竞赛中常见：把 test 也合并进来以对齐处理；标准化后把 NaN 填 0
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features = all_features.astype(float)

    n_train = train_df.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(
        train_df.SalePrice.values.reshape(-1, 1), dtype=torch.float32
    )
    return train_features, test_features, train_labels, test_df


def get_net(num_features: int) -> nn.Module:
    """最简线性模型：Linear(num_features -> 1)。"""
    return nn.Sequential(nn.Linear(num_features, 1))


def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    """对数 RMSE（与 D2L 一致）：在 log 空间算 MSE，再开方。"""
    # 把预测值限制在 [1, +∞)，避免 log(<=0)
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(LOSS_FN(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(
    net: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    valid_features: Optional[torch.Tensor],
    valid_labels: Optional[torch.Tensor],
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
):
    train_ls: List[float] = []
    valid_ls: List[float] = []
    train_iter = load_array((train_features, train_labels), batch_size=batch_size)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = LOSS_FN(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if valid_labels is not None and valid_features is not None:
            valid_ls.append(log_rmse(net, valid_features, valid_labels))
    return train_ls, valid_ls


# k折交叉验证，得到第i折的训练数据和验证数据
def get_k_fold_data(
    k: int, i: int, X: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(
    k: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    *,
    plot_file: str = PLOT_FILE,
) -> Tuple[float, float]:
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # 每次都是从头开始训练
        net = get_net(num_features=X_train.shape[1])
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size
        )
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plot(
                list(range(1, num_epochs + 1)),
                [train_ls, valid_ls],
                xlabel="epoch",
                ylabel="rmse",
                xlim=[1, num_epochs],
                legend=["train", "valid"],
                yscale="log",
                save_name=plot_file,
            )
        print(f"fold {i + 1}, train rmse {train_ls[-1]:f}, valid rmse {valid_ls[-1]:f}")
    return train_l_sum / k, valid_l_sum / k


# 通过k折交叉验证得到最优的超参数，然后进入真正的模型训练，并预测测试集
def train_and_pred(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_df: pd.DataFrame,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    *,
    plot_file: str = PLOT_FILE,
    submission_csv: str = SUBMISSION_CSV,
):
    net = get_net(num_features=train_features.shape[1])
    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        None,
        None,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
    plot(
        list(range(1, num_epochs + 1)),
        [train_ls],
        xlabel="epoch",
        ylabel="log rmse",
        xlim=[1, num_epochs],
        legend=["train"],
        yscale="log",
        save_name=plot_file,
    )
    print(f"train log rmse {float(train_ls[-1]):f}")
    # 使用训练好的模型预测测试集
    test_pred = net(test_features).detach().numpy()
    submission = pd.concat(
        [test_df["Id"], pd.Series(test_pred.reshape(1, -1)[0], name="SalePrice")],
        axis=1,
    )
    submission.to_csv(submission_csv, index=False)


def main() -> None:
    train_features, test_features, train_labels, test_df = load_and_preprocess()

    k = DEFAULT_K
    num_epochs = DEFAULT_NUM_EPOCHS
    lr = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY
    batch_size = DEFAULT_BATCH_SIZE

    # 可选：先用 K 折验证评估当前超参数（用于调参）
    # train_l, valid_l = k_fold(
    #     k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size
    # )
    # print(f"{k}-fold validation: avg train rmse {train_l:f}, avg valid rmse {valid_l:f}")

    # 用选定超参数训练全量训练集，并生成提交文件
    train_and_pred(
        train_features,
        test_features,
        train_labels,
        test_df,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )


if __name__ == "__main__":
    main()
