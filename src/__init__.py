"""
Neural Network from Scratch
===========================

一个纯 NumPy 实现的神经网络框架，从零构建深度学习模型。

模块概览:
- layers: 神经网络层 (Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNorm)
- activations: 激活函数 (ReLU, Sigmoid, Tanh, Softmax, GELU, ELU, Swish, LeakyReLU)
- losses: 损失函数 (MSE, CrossEntropy, BinaryCrossEntropy, Huber, Hinge)
- optimizers: 优化器 (SGD, Momentum, Adam, RMSprop, AdaGrad, AdaDelta) + LRScheduler
- model: 模型容器 (Sequential), 早停(EarlyStopping), 训练历史(History)

快速开始:
    >>> from src import Sequential, Dense, ReLU, Softmax
    >>> from src import Adam, CrossEntropy
    >>>
    >>> model = Sequential()
    >>> model.add(Dense(784, 128))
    >>> model.add(ReLU())
    >>> model.add(Dense(128, 10))
    >>> model.add(Softmax())
    >>>
    >>> history = model.fit(X_train, y_train,
    ...                     epochs=20,
    ...                     batch_size=32,
    ...                     optimizer=Adam(learning_rate=0.001),
    ...                     loss_fn=CrossEntropy(),
    ...                     validation_data=(X_val, y_val))
"""

# 层
from src.layers import (
    Layer,
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

# 激活函数
from src.activations import (
    Activation,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    ELU,
    Swish,
)

# 损失函数
from src.losses import (
    Loss,
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
    Huber,
    Hinge,
)

# 优化器
from src.optimizers import (
    Optimizer,
    SGD,
    Momentum,
    RMSprop,
    Adam,
    AdaGrad,
    AdaDelta,
    LRScheduler,
)

# 模型与工具
from src.model import (
    Sequential,
    EarlyStopping,
    History,
)

__version__ = "1.0.0"
__author__ = "c2318456590-Leo"
