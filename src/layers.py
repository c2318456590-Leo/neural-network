"""
神经网络层模块

包含所有可训练层和操作层：
- Dense: 全连接层（支持权重初始化策略）
- Conv2D: 二维卷积层（基于 im2col 实现）
- MaxPool2D: 最大池化层
- Flatten: 展平层（用于 CNN → 全连接的过渡）
- Dropout: 随机失活层（正则化）
- BatchNormalization: 批归一化层（加速收敛）
"""

import numpy as np


class Layer:
    """所有层的基类，定义统一的前向/反向传播接口"""
    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


# ============================================================
# 全连接层 (Fully Connected / Dense Layer)
# ============================================================
class Dense(Layer):
    """
    全连接层: output = x @ W + b

    参数:
        input_size: 输入特征维度
        output_size: 输出特征维度
        weight_init: 权重初始化方式 ('he', 'xavier', 'normal')
    """
    def __init__(self, input_size, output_size, weight_init='he'):
        self.input_size = input_size
        self.output_size = output_size

        # 根据策略初始化权重
        if weight_init == 'he':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # 梯度缓存
        self.dW = None
        self.db = None

    def forward(self, x, training=True):
        """前向传播: y = xW + b"""
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        """
        反向传播:
        dx = dout @ W.T
        dW = x.T @ dout
        db = sum(dout)
        """
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)


# ============================================================
# 卷积层 (Convolutional 2D Layer)
# ============================================================
class Conv2D(Layer):
    """
    二维卷积层，使用 im2col 技巧将卷积转化为矩阵乘法

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数（滤波器数量）
        kernel_size: 卷积核大小（int 或 tuple）
        stride: 步长
        padding: 填充大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化
        fan_in = in_channels * self.kernel_h * self.kernel_w
        self.W = np.random.randn(out_channels, in_channels,
                                  self.kernel_h, self.kernel_w) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((1, out_channels))

        self.dW = None
        self.db = None

    def _im2col(self, x, kH, kW, stride):
        """将输入图像转换为列矩阵（im2col）以便矩阵运算"""
        N, C, H, W = x.shape
        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1

        col = np.zeros((N, C, kH, kW, out_h, out_w))
        for i in range(kH):
            i_max = i + stride * out_h
            for j in range(kW):
                j_max = j + stride * out_w
                col[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col, out_h, out_w

    def _col2im(self, col, input_shape, kH, kW, stride):
        """将列矩阵还原为图像格式（col2im）"""
        N, C, H, W = input_shape
        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1
        col = col.reshape(N, out_h, out_w, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros(input_shape)
        for i in range(kH):
            i_max = i + stride * out_h
            for j in range(kW):
                j_max = j + stride * out_w
                img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
        return img

    def _pad(self, x, pad):
        """对输入进行零填充"""
        if pad > 0:
            return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                           mode='constant', constant_values=0)
        return x

    def forward(self, x, training=True):
        """前向传播: 使用 im2col 进行高效卷积"""
        self.original_shape = x.shape
        N, C, H, W = x.shape

        if self.padding > 0:
            x_padded = self._pad(x, self.padding)
        else:
            x_padded = x

        self.padded_shape = x_padded.shape  # 保存填充后的形状用于反向传播
        self.col, out_h, out_w = self._im2col(
            x_padded, self.kernel_h, self.kernel_w, self.stride)

        # 将滤波器展平为二维矩阵
        W_col = self.W.reshape(self.out_channels, -1).T

        # 卷积 = 矩阵乘法
        out = np.dot(self.col, W_col) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x_col = self.col
        return out

    def backward(self, dout):
        """反向传播: 计算关于输入、权重和偏置的梯度"""
        N, _, out_h, out_w = dout.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        W_col = self.W.reshape(self.out_channels, -1)

        # 梯度
        self.db = np.sum(dout_reshaped, axis=0, keepdims=True)
        self.dW = np.dot(self.x_col.T, dout_reshaped).T.reshape(self.W.shape)

        dcol = np.dot(dout_reshaped, self.W.reshape(self.out_channels, -1))
        dx = self._col2im(dcol, self.padded_shape,
                          self.kernel_h, self.kernel_w, self.stride)

        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx


# ============================================================
# 最大池化层 (Max Pooling 2D)
# ============================================================
class MaxPool2D(Layer):
    """
    二维最大池化层：在每个池化窗口中取最大值

    参数:
        pool_size: 池化窗口大小
        stride: 步长（默认等于 pool_size）
    """
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x, training=True):
        """前向传播: 取每个窗口的最大值"""
        self.x = x
        N, C, H, W = x.shape
        pH = pW = self.pool_size
        s = self.stride

        out_h = (H - pH) // s + 1
        out_w = (W - pW) // s + 1

        out = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros_like(out, dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                window = x[:, :, h_start:h_start+pH, w_start:w_start+pW]
                window_flat = window.reshape(N, C, -1)
                out[:, :, i, j] = np.max(window_flat, axis=2)
                self.max_indices[:, :, i, j] = np.argmax(window_flat, axis=2)

        return out

    def backward(self, dout):
        """反向传播: 只将梯度传递给最大值位置"""
        N, C, H, W = self.x.shape
        pH = pW = self.pool_size
        s = self.stride
        _, _, out_h, out_w = dout.shape

        dx = np.zeros_like(self.x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                idx = self.max_indices[:, :, i, j]
                # 将梯度分配到最大值对应的位置
                n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C),
                                            indexing='ij')
                dx[n_idx, c_idx,
                   h_start + idx // pW,
                   w_start + idx % pW] += dout[:, :, i, j]

        return dx


# ============================================================
# 展平层 (Flatten Layer)
# ============================================================
class Flatten(Layer):
    """
    展平层: 将多维输入展平为一维向量
    主要用于从卷积层到全连接层的过渡
    例如: (N, C, H, W) -> (N, C*H*W)
    """
    def forward(self, x, training=True):
        """保存原始形状并展平"""
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        """恢复原始形状"""
        return dout.reshape(self.original_shape)


# ============================================================
# 随机失活层 (Dropout)
# ============================================================
class Dropout(Layer):
    """
    Dropout 正则化层：训练时随机将部分神经元置零，
    推理时按比例缩放输出以保持期望不变（Inverted Dropout）

    参数:
        rate: 失活概率（0~1之间）
    """
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if training and self.rate > 0:
            self.mask = (np.random.rand(*x.shape) >= self.rate).astype(float)
            return x * self.mask / (1.0 - self.rate)
        return x

    def backward(self, dout):
        if self.rate > 0:
            return dout * self.mask / (1.0 - self.rate)
        return dout


# ============================================================
# 批归一化层 (Batch Normalization)
# ============================================================
class BatchNormalization(Layer):
    """
    批归一化层：对小批量数据进行标准化处理，
    加速网络训练、减少对初始化的敏感性、起到轻微正则化作用

    公式: BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta

    参数:
        num_features: 特征数量
        momentum: 运行均值/方差的动量系数
        epsilon: 数值稳定性的小常数
    """
    def __init__(self, num_features, momentum=0.9, epsilon=1e-7):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.epsilon = epsilon

        # 推理时使用的运行统计量
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # 梯度
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, training=True):
        if training:
            # 计算当前批次的均值和方差
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)

            # 归一化
            self.x_norm = (x - self.batch_mean) / np.sqrt(
                self.batch_var + self.epsilon)

            # 缩放和平移
            out = self.gamma * self.x_norm + self.beta

            # 更新运行统计量（用于推理）
            self.running_mean = (
                self.momentum * self.running_mean +
                (1 - self.momentum) * self.batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var +
                (1 - self.momentum) * self.batch_var
            )
        else:
            # 推理时使用运行统计量
            self.x_norm = (x - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon)
            out = self.gamma * self.x_norm + self.beta

        return out

    def backward(self, dout):
        """BatchNorm 反向传播（链式法则推导）"""
        m = dout.shape[0]

        # gamma 和 beta 的梯度
        self.dgamma = np.sum(dout * self.x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        # 关于输入的梯度
        dx_norm = dout * self.gamma
        std_inv = 1.0 / np.sqrt(self.batch_var + self.epsilon)

        dx = (std_inv / m) * (
            m * dx_norm
            - np.sum(dx_norm, axis=0, keepdims=True)
            - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True)
        )
        return dx
