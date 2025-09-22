"""
XOR 异或问题求解

XOR 是经典的非线性可分问题，单层感知机无法解决，
必须通过隐藏层来学习非线性边界。

这个例子展示了神经网络的"万能近似"能力：
即使只有 2 个输入、4 个数据点，也需要至少一个隐藏层才能正确分类。

网络结构:
  Dense(2->8) -> ReLU -> Dense(8->4) -> ReLU -> Dense(4->1) -> Sigmoid

运行方式:
    python examples/train_xor.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src import Sequential, Dense, ReLU, Sigmoid, Adam, BinaryCrossEntropy


def main():
    print("=" * 50)
    print("  XOR 异或问题求解")
    print("=" * 50)

    # XOR 真值表
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32).reshape(-1, 1)

    print("\n数据:")
    for i in range(4):
        print(f"  {X[i]} -> {int(y[i][0])}")

    # 构建模型（需要隐藏层！）
    model = Sequential()
    model.add(Dense(2, 8))
    model.add(ReLU())
    model.add(Dense(8, 4))
    model.add(ReLU())
    model.add(Dense(4, 1))
    model.add(Sigmoid())

    # 训练
    print("\n训练中...")
    history = model.fit(
        X, y,
        epochs=2000,
        batch_size=4,
        optimizer=Adam(learning_rate=0.1),
        loss_fn=BinaryCrossEntropy(),
        verbose=False,
    )

    # 预测
    predictions = model.predict_proba(X)
    predicted_classes = (predictions > 0.5).astype(int)

    print("\n结果:")
    print(f"{'输入':<12s} {'真实':<6s} {'预测概率':<12s} {'预测':<6s} {'状态'}")
    print("-" * 50)
    for i in range(4):
        status = "✓ 正确" if predicted_classes[i][0] == int(y[i][0]) else "✗ 错误"
        print(f"{str(X[i]):<12s} {int(y[i][0]):<6d} "
              f"{predictions[i][0]:<12.4f} {predicted_classes[i][0]:<6d} {status}")

    final_loss = history.train_losses[-1]
    accuracy = np.mean(predicted_classes.flatten() == y.flatten())
    print(f"\n最终损失: {final_loss:.6f}")
    print(f"准确率: {accuracy*100:.1f}%")

    # 可视化决策边界
    _plot_decision_boundary(model, X, y.flatten())


def _plot_decision_boundary(model, X, y):
    """绘制二维决策边界"""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    step = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid).flatten()
    zz = probs.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, zz, levels=50, cmap='RdBu_r', alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)

    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='k',
               linewidths=2, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('XOR 决策边界 (Neural Network)')
    fig.colorbar(contour, label='P(y=1)')

    save_path = os.path.join(os.path.dirname(__file__), 'xor_decision_boundary.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n决策边界图已保存: {save_path}")


if __name__ == '__main__':
    main()
