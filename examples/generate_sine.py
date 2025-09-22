"""
正弦波序列预测 (Simple RNN-style Demo)

使用全连接网络实现简单的序列预测任务：
给定前 n 个时间步的正弦波值，预测下一个时间步的值。

这展示了神经网络处理时序/回归问题的能力。

网络结构:
  Dense(window_size->64) -> Tanh -> Dense(64->32) -> ReLU -> Dense(32->1)

运行方式:
    python examples/generate_sine.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src import Sequential, Dense, ReLU, Tanh, Adam, MSE


def generate_sine_data(n_samples=1000, window_size=20, freq=1.0):
    """
    生成正弦波滑动窗口数据

    参数:
        n_samples: 生成的数据点数
        window_size: 输入窗口大小（用前 N 个点预测第 N+1 个点）
        freq: 正弦波频率
    返回:
        X: (n_samples, window_size) 输入序列
        y: (n_samples,) 目标值
    """
    t = np.linspace(0, n_samples * 0.05, n_samples + window_size)
    sine_wave = np.sin(2 * np.pi * freq * t)

    X = []
    y = []
    for i in range(n_samples):
        X.append(sine_wave[i:i + window_size])
        y.append(sine_wave[i + window_size])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    print("=" * 55)
    print("  正弦波序列预测 (回归任务)")
    print("=" * 55)

    # 生成数据
    window_size = 15
    X, y = generate_sine_data(n_samples=800, window_size=window_size, freq=1.0)

    # 划分训练/测试
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n数据: 总计{len(X)} 样本 | 窗口大小={window_size}")
    print(f"训练: {len(X_train)} | 测试: {len(X_test)}")

    # 构建回归模型
    model = Sequential()
    model.add(Dense(window_size, 64))
    model.add(Tanh())
    model.add(Dense(64, 32))
    model.add(ReLU())
    model.add(Dense(32, 1))

    model.summary()

    # 训练（MSE 回归损失）
    # 注意: y 需要 reshape 为 (N, 1) 以匹配输出维度
    history = model.fit(
        X_train, y_train.reshape(-1, 1),
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test.reshape(-1, 1)),
        optimizer=Adam(learning_rate=0.005),
        loss_fn=MSE(),
        verbose=True,
    )

    # 预测并可视化
    y_pred = model.predict_proba(X_test).flatten()

    # 计算指标
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))

    print(f"\n测试集指标:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # 绘制结果
    _plot_results(y_test, y_pred, history)


def _plot_results(y_true, y_pred, history):
    """绘制预测曲线和训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 预测 vs 真实值
    axes[0].plot(y_true, label='True', alpha=0.7, linewidth=1.5)
    axes[0].plot(y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Sine Wave Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 训练损失曲线
    axes[1].plot(history.train_losses, label='Train Loss', color='blue')
    if history.val_losses:
        axes[1].plot(history.val_losses, label='Val Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Training Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'sine_prediction.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果图已保存: {save_path}")


if __name__ == '__main__':
    main()
