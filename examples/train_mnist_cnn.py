"""
CNN 训练 MNIST 手写数字识别

演示如何使用卷积神经网络(Conv2D + MaxPool2D + Dense)进行图像分类。
这是深度学习中最经典的入门任务之一。

网络结构:
  Input(1,28,28) -> Conv2D(8) -> ReLU -> MaxPool2D -> Conv2D(16) -> ReLU
  -> MaxPool2D -> Flatten -> Dense(128) -> ReLU -> Dense(10) -> Softmax

运行方式:
    python examples/train_mnist_cnn.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import (
    Sequential, Conv2D, MaxPool2D, Flatten, Dense, Dropout,
    ReLU, Softmax, Adam, CrossEntropy, EarlyStopping, LRScheduler
)


def load_mnist():
    """加载并预处理 MNIST 数据集"""
    print("正在加载 MNIST 数据集...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # 归一化到 [0, 1]
    X = X.astype(np.float32) / 255.0

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # 重塑为图像格式 (N, C, H, W)，MNIST 是单通道灰度图
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    print(f"训练集: {X_train.shape[0]} 张 | 测试集: {X_test.shape[0]} 张")
    return X_train, X_test, y_train, y_test


def build_cnn():
    """构建 CNN 模型"""
    model = Sequential()

    # 第一个卷积块: 提取低级特征（边缘、线条）
    model.add(Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2))   # 28x28 -> 14x14

    # 第二个卷积块: 提取高级特征（形状、纹理）
    model.add(Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2))   # 14x14 -> 7x7

    # 分类头
    model.add(Flatten())                # 16*7*7 = 784
    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, 10))
    model.add(Softmax())

    return model


def main():
    print("=" * 60)
    print("  CNN 手写数字识别 - MNIST")
    print("=" * 60)
    print()

    # 加载数据
    X_train, X_test, y_train, y_test = load_mnist()
    print()

    # 构建模型
    model = build_cnn()
    model.summary()

    # 配置训练组件
    optimizer = Adam(learning_rate=0.005)
    loss_fn = CrossEntropy()
    early_stop = EarlyStopping(patience=5, min_delta=1e-4)
    lr_scheduler = LRScheduler(optimizer, mode='step', step_size=5, gamma=0.5)

    # 训练
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_test, y_test),
        optimizer=optimizer,
        loss_fn=loss_fn,
        early_stopping=early_stop,
        lr_scheduler=lr_scheduler,
        verbose=True,
    )

    # 最终评估
    print("\n最终评估:")
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)
    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 展示一些预测结果
    print("\n预测样例 (前10个测试样本):")
    predictions = model.predict(X_test[:10])
    for i in range(10):
        result = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  预测={predictions[i]} | 实际={y_test[i]} {result}")


if __name__ == '__main__':
    main()
