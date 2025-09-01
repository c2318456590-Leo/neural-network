"""
激活函数模块

包含常用的神经网络激活函数，每个函数都实现了前向传播和反向传播：
- ReLU: 修正线性单元（最常用）
- LeakyReLU: 带泄漏的 ReLU（解决神经元死亡问题）
- Sigmoid: S 形激活函数（输出概率时使用）
- Tanh: 双曲正切函数（输出范围 -1 到 1）
- Softmax: 软最大值（多分类输出层）
- GELU: 高斯误差线性单元（Transformer 中常用）
- ELU: 指数线性单元
- Swish (SiLU): 自门控激活函数
"""

import numpy as np


class Activation:
    """所有激活函数的基类"""
    def forward(self, x, **kwargs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


# ============================================================
# ReLU (Rectified Linear Unit)
# ============================================================
class ReLU(Activation):
    """
    修正线性单元: f(x) = max(0, x)
    特点：计算简单、缓解梯度消失、稀疏激活
    """
    def forward(self, x, **kwargs):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, dout):
        """梯度: f'(x) = 1 if x > 0 else 0"""
        dx = dout.copy()
        dx[self.output <= 0] = 0
        return dx


# ============================================================
# LeakyReLU
# ============================================================
class LeakyReLU(Activation):
    """
    带泄漏的修正线性单元:
      f(x) = x          if x > 0
      f(x) = alpha * x   if x <= 0
    解决 ReLU 的"神经元死亡"问题
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, **kwargs):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        """梯度: f'(x) = 1 if x > 0 else alpha"""
        return np.where(self.x > 0, dout, self.alpha * dout)


# ============================================================
# Sigmoid
# ============================================================
class Sigmoid(Activation):
    """
    Sigmoid 激活函数: f(x) = 1 / (1 + exp(-x))
    输出范围 (0, 1)，常用于二分类输出层
    """
    def forward(self, x, **kwargs):
        # 数值裁剪防止溢出
        self.output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        return self.output

    def backward(self, dout):
        """梯度: f'(x) = sigmoid(x) * (1 - sigmoid(x))"""
        return dout * self.output * (1 - self.output)


# ============================================================
# Tanh
# ============================================================
class Tanh(Activation):
    """
    双曲正切函数: f(x) = tanh(x)
    输出范围 (-1, 1)，零中心化，比 Sigmoid 收敛更快
    """
    def forward(self, x, **kwargs):
        self.output = np.tanh(x)
        return self.output

    def backward(self, dout):
        """梯度: f'(x) = 1 - tanh^2(x)"""
        return dout * (1 - self.output ** 2)


# ============================================================
# Softmax
# ============================================================
class Softmax(Activation):
    """
    Softmax 函数: 将向量转换为概率分布
    softmax(x_i) = exp(x_i) / sum(exp(x_j))
    常用于多分类问题的输出层
    """
    def forward(self, x, **kwargs):
        # 数值稳定性：减去最大值防止溢出
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dout):
        """
        当 Softmax 与交叉熵损失配合使用时，
        反向传播简化为: dL/dz = y_pred - y_true
        因此这里直接返回 dout（已由损失函数处理）
        """
        return dout


# ============================================================
# GELU (Gaussian Error Linear Unit)
# ============================================================
class GELU(Activation):
    """
    高斯误差线性单元: f(x) = x * Phi(x)
    其中 Phi 是标准正态分布的累积分布函数(CDF)

    近似公式: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Transformer (BERT, GPT) 中广泛使用的激活函数
    """
    def forward(self, x, **kwargs):
        self.x = x
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
        self.output = 0.5 * x * (1.0 + np.tanh(inner))
        return self.output

    def backward(self, dout):
        """基于 tanh 近似的 GELU 导数"""
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (self.x + 0.044715 * self.x ** 3)
        tanh_inner = np.tanh(inner)
        # dGELU/dx ≈ 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d(inner)/dx
        d_inner = sqrt_2_over_pi * (1 + 0.134145 * self.x ** 2)
        derivative = 0.5 * (1 + tanh_inner) + 0.5 * self.x * (1 - tanh_inner ** 2) * d_inner
        return dout * derivative


# ============================================================
# ELU (Exponential Linear Unit)
# ============================================================
class ELU(Activation):
    """
    指数线性单元:
      f(x) = x           if x > 0
      f(x) = alpha*(exp(x)-1) if x <= 0
    特点：负区间有非零输出，均值接近零
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x, **kwargs):
        self.x = x
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -50, 0)) - 1))

    def backward(self, dout):
        """梯度: f'(x) = 1 if x > 0 else f(x) + alpha"""
        output = np.where(
            self.x > 0, self.x,
            self.alpha * (np.exp(np.clip(self.x, -50, 0)) - 1)
        )
        return np.where(self.x > 0, dout, dout * (output + self.alpha))


# ============================================================
# Swish (SiLU)
# ============================================================
class Swish(Activation):
    """
    Swish 激活函数: f(x) = x * sigmoid(x)
    也称为 SiLU (Sigmoid Linear Unit)
    特点：自门控特性，在深层网络中表现优于 ReLU
    """
    def forward(self, x, **kwargs):
        self.x = x
        self.sigmoid_x = self._sigmoid(x)
        self.output = x * self.sigmoid_x
        return self.output

    def backward(self, dout):
        """梯度: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
        derivative = self.sigmoid_x + self.x * self.sigmoid_x * (1 - self.sigmoid_x)
        return dout * derivative

    @staticmethod
    def _sigmoid(x):
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
