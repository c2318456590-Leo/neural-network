"""
损失函数模块

用于衡量模型预测与真实标签之间的差异，是反向传播的起点：

- MSE: 均方误差（回归任务）
- CrossEntropy: 交叉熵损失（多分类任务，配合 Softmax 使用）
- BinaryCrossEntropy: 二分类交叉熵（二分类任务，配合 Sigmoid 使用）
- Huber: Huber 损失（对异常值鲁棒的回归损失）
- Hinge: Hinge 损失（支持向量机风格的最大间隔损失）
"""

import numpy as np


class Loss:
    """所有损失函数的基类"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# ============================================================
# MSE (Mean Squared Error)
# ============================================================
class MSE(Loss):
    """
    均方误差损失: L = mean((y_pred - y_true)^2)
    用于回归任务
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        m = y_pred.shape[0]
        self.loss = np.mean((y_pred - y_true) ** 2)
        return self.loss

    def backward(self):
        """梯度: dL/dy_pred = 2/n * (y_pred - y_true)"""
        m = self.y_pred.shape[0]
        return (2.0 / m) * (self.y_pred - self.y_true)


# ============================================================
# Cross Entropy Loss (for multi-class classification)
# ============================================================
class CrossEntropy(Loss):
    """
    交叉熵损失（多分类）:
      L = -mean(sum(y_true * log(y_pred)))
    配合 Softmax 输出层使用。
    支持两种格式:
      - 整数标签: y_true 为类别索引 (N,)
      - one-hot 编码: y_true 为 (N, C) 的矩阵
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        m = y_pred.shape[0]

        # 自动检测标签格式
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # 整数标签格式
            y_true_flat = y_true.flatten().astype(int)
            log_likelihood = -np.log(
                y_pred[np.arange(m), y_true_flat] + 1e-12
            )
        else:
            # One-hot 编码格式
            log_likelihood = -np.sum(
                y_true * np.log(y_pred + 1e-12), axis=1
            )

        self.loss = np.mean(log_likelihood)
        return self.loss

    def backward(self):
        """
        Softmax + CrossEntropy 组合的反向传播简化为:
        dL/dz = (y_pred - y_true) / N
        """
        m = self.y_pred.shape[0]
        dx = self.y_pred.copy()

        if self.y_true.ndim == 1 or self.y_true.shape[1] == 1:
            y_true_flat = self.y_true.flatten().astype(int)
            dx[np.arange(m), y_true_flat] -= 1
        else:
            dx -= self.y_true

        dx /= m
        return dx


# ============================================================
# Binary Cross Entropy Loss
# ============================================================
class BinaryCrossEntropy(Loss):
    """
    二分类交叉熵损失:
      L = -mean(y*log(p) + (1-y)*log(1-p))
    配合 Sigmoid 输出层使用，用于二分类任务
    """
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        self.y_true = y_true
        m = y_pred.shape[0]

        self.loss = -np.mean(
            y_true * np.log(self.y_pred) +
            (1 - y_true) * np.log(1 - self.y_pred)
        )
        return self.loss

    def backward(self):
        """梯度: dL/dp = -(y/p - (1-y)/(1-p)) / N"""
        m = self.y_pred.shape[0]
        dx = -(self.y_true / self.y_pred -
               (1 - self.y_true) / (1 - self.y_pred)) / m
        return dx


# ============================================================
# Huber Loss
# ============================================================
class Huber(Loss):
    """
    Huber 损失: 结合了 MSE 和 MAE 的优点
      L = 0.5 * (y - f)^2              if |y-f| <= delta
      L = delta * |y-f| - 0.5*delta^2  otherwise
    对异常值比 MSE 更鲁棒，比 MAE 处处可微

    参数:
        delta: 分界阈值（默认 1.0）
    """
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        error = y_pred - y_true
        abs_error = np.abs(error)
        is_small_error = abs_error <= self.delta

        # 小误差用平方损失，大误差用绝对值损失
        quadratic = 0.5 * error ** 2
        linear = self.delta * abs_error - 0.5 * self.delta ** 2

        self.loss = np.mean(
            np.where(is_small_error, quadratic, linear)
        )
        return self.loss

    def backward(self):
        """梯度: 在 delta 处平滑过渡"""
        error = self.y_pred - self.y_true
        abs_error = np.abs(error)
        is_small_error = abs_error <= self.delta

        grad_quadratic = error
        grad_linear = self.delta * np.sign(error)

        return np.mean(
            np.where(is_small_error, grad_quadratic, grad_linear),
            axis=0, keepdims=True
        )


# ============================================================
# Hinge Loss
# ============================================================
class Hinge(Loss):
    """
    Hinge 损失: L = max(0, 1 - y_true * y_pred)
    用于支持向量机(SVM)风格的分类任务
    鼓励正确类别的分数至少比错误类别高一个 margin

    参数:
        margin: 间隔大小（默认 1.0）
    """
    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        # 确保 y_true 为 {-1, +1}
        if y_true.max() <= 1:
            y_signed = 2 * y_true - 1
        else:
            y_signed = y_true

        hinge_loss = np.maximum(0, self.margin - y_signed * y_pred)
        self.loss = np.mean(hinge_loss)
        return self.loss

    def backward(self):
        """梯度: 只对违反间隔约束的样本有非零梯度"""
        if self.y_true.max() <= 1:
            y_signed = 2 * self.y_true - 1
        else:
            y_signed = self.y_true

        margin_violation = (self.margin - y_signed * self.y_pred) > 0
        return -np.mean(margin_violation * y_signed, axis=0, keepdims=True)
