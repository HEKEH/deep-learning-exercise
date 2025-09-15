"""
Tools包 - 深度学习工具函数集合

这个包包含了常用的深度学习工具函数，包括：
- 数据加载函数
- 训练和评估函数
- 可视化工具
- 实用工具类
"""

# 导入所有常用的函数和类
from .tools import (
    # 数据加载
    load_data_fashion_mnist,

    # 训练和评估
    train_ch3,
    train_epoch_ch3,
    evaluate_accuracy,
    accuracy,

    # 预测和可视化
    predict_ch3,

    # 工具类
    Accumulator,
    Animator,

    # 配置函数
    setup_matplotlib
)

# 定义__all__，控制from tools import *的行为
__all__ = [
    'load_data_fashion_mnist',
    'train_ch3',
    'train_epoch_ch3',
    'evaluate_accuracy',
    'accuracy',
    'predict_ch3',
    'Accumulator',
    'Animator',
    'setup_matplotlib'
]

# 包版本信息
__version__ = '1.0.0'
__author__ = 'Will'