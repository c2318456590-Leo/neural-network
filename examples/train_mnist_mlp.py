"""
MLP 训练 MNIST 手写数字识别

使用多层感知机(全连接网络)进行 MNIST 分类，
演示早停机制和学习率调度器的使用。

网络结构:
  Dense(784->512) -> ReLU -> BatchNorm -> Dropout ->
  Dense(512->256) -> ReLU -> BatchNorm ->
  Dense(256->128) -> ReLU ->
  Dense(128->10) -> Softmax

运行方式:
    python examples/train_mnist_mlp.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import (
    Sequential, Dense, Dropout, BatchNormalization,
    ReLU, Softmax, Adam, CrossEntropy, EarlyStopping, LRScheduler
)


def main():
    print("=" * 60)
    print("  MLP 手写数字识别 - MNIST")
    print("=" * 60)
    print()

    # 加载数据
    print("正在加载 MNIST 数据集...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    X = X.astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"训练: {X_train.shape[0]} | 验证: {X_val.shape[0]}")
    print()

    # 构建模型
    model = Sequential()
    model.add(Dense(784, 512))
    model.add(ReLU())
    model.add(BatchNormalization(num_features=512))
    model.add(Dropout(rate=0.2))

    model.add(Dense(512, 256))
    model.add(ReLU())
    model.add(BatchNormalization(num_features=256))

    model.add(Dense(256, 128))
    model.add(ReLU())

    model.add(Dense(128, 10))
    model.add(Softmax())

    model.summary()

    # 训练配置
    opt = Adam(learning_rate=0.002)
    loss_fn = CrossEntropy()

    # 早停：验证损失连续 5 轮不下降则停止
    early_stop = EarlyStopping(patience=5, min_delta=1e-4)

    # 学习率调度：余弦退火
    lr_sched = LRScheduler(opt, mode='cosine', max_epochs=30, min_lr=1e-5)

    # 训练
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        optimizer=opt,
        loss_fn=loss_fn,
        early_stopping=early_stop,
        lr_scheduler=lr_sched,
        verbose=True,
    )

    # 结果
    print(f"\n最终验证 Loss: {history.val_losses[-1]:.4f}")
    print(f"最终验证准确率: {history.val_accuracies[-1]*100:.2f}%")
    print(f"训练总轮数: {len(history.train_losses)}")


if __name__ == '__main__':
    main()
