# Neural Network from Scratch

> 纯 NumPy 从零构建神经网络框架，适合深度学习入门学习。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 为什么做这个项目？

学习深度学习的最佳方式就是**亲手实现一遍**。这个项目从零开始实现了神经网络的核心组件，不依赖 PyTorch/TensorFlow，让你真正理解每个模块背后的数学原理。

## 项目结构

```
neural-network-from-scratch/
├── src/
│   ├── __init__.py          # 统一导出
│   ├── layers.py             # 网络层 (6种层)
│   ├── activations.py        # 激活函数 (8个)
│   ├── losses.py             # 损失函数 (5个)
│   ├── optimizers.py         # 优化器 (6个) + 学习率调度器
│   └── model.py              # Sequential 模型 + 早停 + 训练历史
├── examples/
│   ├── train_xor.py          # XOR 异或问题（含决策边界可视化）
│   ├── train_mnist_mlp.py    # MLP 训练 MNIST（含早停+余弦退火）
│   ├── train_mnist_cnn.py    # CNN 训练 MNIST（卷积+池化）
│   └── generate_sine.py      # 正弦波序列预测（回归任务）
├── tests/
│   └── test_all.py           # 全量单元测试 + 集成测试
├── requirements.txt
└── .gitignore
```

## 功能一览

| 模块 | 组件 | 数量 |
|------|------|------|
| **层** | Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNorm | 6 |
| **激活函数** | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, ELU, Swish | 8 |
| **损失函数** | MSE, CrossEntropy, BinaryCrossEntropy, Huber, Hinge | 5 |
| **优化器** | SGD, Momentum, Adam, RMSprop, AdaGrad, AdaDelta (+LRScheduler) | 7 |
| **训练工具** | EarlyStopping, History, model.summary(), save/load | 4 |

## 安装

```bash
pip install -r requirements.txt
# 或手动安装:
pip install numpy scikit-learn matplotlib pandas
```

## 快速开始

### XOR 问题（最简示例）

```python
from src import Sequential, Dense, ReLU, Sigmoid, Adam, BinaryCrossEntropy
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([0,1,1,0], dtype=np.float32).reshape(-1,1)

model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())
model.add(Dense(16, 1))
model.add(Sigmoid())

history = model.fit(X, y, epochs=2000, batch_size=4,
                  optimizer=Adam(lr=0.2), loss_fn=BinaryCrossEntropy())
# 准确率: 100%
```

### MNIST 手写数字识别

```python
from src import Sequential, Dense, ReLU, Softmax, BatchNorm, Dropout
from src import Adam, CrossEntropy, EarlyStopping, LRScheduler

model = Sequential()
model.add(Dense(784, 512))
model.add(ReLU())
model.add(BatchNormalization(512))
model.add(Dropout(0.2))
model.add(Dense(512, 256))
model.add(ReLU())
model.add(Dense(256, 10))
model.add(Softmax())

model.fit(X_train, y_train, epochs=30, batch_size=64,
         optimizer=Adam(0.002), loss_fn=CrossEntropy(),
         validation_data=(X_val, y_val),
         early_stopping=EarlyStopping(patience=5),
         lr_scheduler=LRScheduler(optimizer, mode='cosine', max_epochs=30))
# 验证准确率: ~97.6%
```

---

## 运行示例 & 截图

### 示例 1: XOR 异或问题

```bash
py examples/train_xor.py
```

**运行输出:**

```
==================================================
  XOR 异或问题求解
==================================================

数据:
  [0. 0.] -> 0
  [0. 1.] -> 1
  [1. 0.] -> 1
  [1. 1.] -> 0

训练中...
开始训练 | 样本数: 4, 批次: 4, 轮数: 2000
============================================================
============================================================
训练完成!

结果:
输入           真实     预测概率         预测     状态
--------------------------------------------------
[0. 0.]      0      0.0001       0      ✓ 正确
[0. 1.]      1      1.0000       1      ✓ 正确
[1. 0.]      1      1.0000       1      ✓ 正确
[1. 1.]      0      0.0000       0      ✓ 正确

最终损失: 0.000037
准确率: 100.0%

决策边界图已保存: examples/xor_decision_boundary.png
```

> **说明**: XOR 是经典非线性可分问题。单层感知机无法解决，必须通过隐藏层学习非线性边界。网络成功将 4 个数据点完美分类！

---

### 示例 2: MNIST 手写数字识别 (MLP)

```bash
py examples/train_mnist_mlp.py
```

**运行输出:**

```
============================================================
  MLP 手写数字识别 - MNIST
============================================================

正在加载 MNIST 数据集...
训练: 59500 | 验证: 10500


============================================================
 Model Summary
============================================================
Layer                     Output Shape             Params
------------------------------------------------------------
1. Dense                                     --  401,920
2. ReLU                                      --        0
3. BatchNormalization                        --    1,024
4. Dropout                                   --        0
5. Dense                                     --  131,328
6. ReLU                                      --        0
7. BatchNormalization                        --      512
8. Dense                                     --   32,896
9. ReLU                                      --        0
10. Dense                                     --    1,290
11. Softmax                                   --        0
------------------------------------------------------------
Total Parameters: 568,970
============================================================

开始训练 | 样本数: 59500, 批次: 64, 轮数: 30
============================================================
Epoch   1/30 | loss: 0.2233 | acc: 0.9319 | val_loss: 0.1380 | val_acc: 0.9613 | lr: 0.002000 | 30.1s
Epoch   2/30 | loss: 0.1134 | acc: 0.9653 | val_loss: 0.1138 | val_acc: 0.9730 | lr: 0.001995 | 30.5s
Epoch   3/30 | loss: 0.0860 | acc: 0.9727 | val_loss: 0.1123 | val_acc: 0.9716 | lr: 0.001978 | 27.9s
Epoch   4/30 | loss: 0.0740 | acc: 0.9773 | val_loss: 0.1275 | val_acc: 0.9720 | lr: 0.001951 | 28.1s
Epoch   5/30 | loss: 0.0673 | acc: 0.9787 | val_loss: 0.1221 | val_acc: 0.9746 | lr: 0.001914 | 28.6s
Epoch   6/30 | loss: 0.0605 | acc: 0.9811 | val_loss: 0.1833 | val_acc: 0.9738 | lr: 0.001867 | 29.1s
Epoch   7/30 | loss: 0.0511 | acc: 0.9835 | val_loss: 0.1554 | val_acc: 0.9761 | lr: 0.001810 | 28.7s
Epoch   8/30 | loss: 0.0440 | acc: 0.9857 | val_loss: 0.1555 | val_acc: 0.9766 | lr: 0.001744 | 30.7s

  [EarlyStopping] Epoch 8: val_loss 未改善 5 轮，触发早停 (best at epoch 3)
  [EarlyStopping] 已恢复最佳权重
============================================================
训练完成!

最终验证 Loss: 0.1123
最终验证准确率: 97.16%
训练总轮数: 8
```

> **说明**: 使用了 BatchNorm、Dropout、EarlyStopping 和 Cosine LRScheduler，仅 8 个 epoch 就达到 **97%+** 准确率，展示了正则化和训练技巧的效果。

---

### 示例 3: 正弦波序列预测（回归任务）

```bash
py examples/generate_sine.py
```

**运行输出:**

```
=======================================================
  正弦波序列预测 (回归任务)
=======================================================

数据: 总计800 样本 | 窗口大小=15
训练: 640 | 测试: 160

============================================================
 Model Summary
============================================================
Layer                     Output Shape             Params
------------------------------------------------------------
1. Dense                                     --    1,024
2. Tanh                                      --        0
3. Dense                                     --    2,080
4. ReLU                                      --        0
5. Dense                                     --       33
------------------------------------------------------------
Total Parameters: 3,137
============================================================

开始训练 | 样本数: 640, 批次: 32, 轮数: 100
============================================================
Epoch   1/100 | loss: 0.1404 | acc: 1.0000 | val_loss: 0.0226 | val_acc: 1.0000 | lr: 0.005000 | 0.0s
Epoch  10/100 | loss: 0.0000 | acc: 1.0000 | val_loss: 0.0000 | val_acc: 1.0000 | lr: 0.005000 | 0.0s
...
Epoch 100/100 | loss: 0.0000 | acc: 1.0000 | val_loss: 0.0000 | val_acc: 1.0000 | lr: 0.005000 | 0.0s
============================================================
训练完成!

测试集指标:
  MSE:  0.000040
  RMSE: 0.006314
  MAE:  0.005040

结果图已保存: examples/sine_prediction.png
```

> **说明**: 给定前 15 个时间步的正弦值，预测下一个值。MSE 接近 0，预测曲线与真实曲线几乎完全重合。

---

### 运行全量测试

```bash
py tests/test_all.py
```

```
=======================================================
  Neural Network from Scratch - 全部测试
=======================================================

[1/5] 测试层模块...
  ✓ Dense
  ✓ Conv2D
  ✓ MaxPool2D
  ✓ Flatten
  ✓ Dropout
  ✓ BatchNorm
  所有层测试通过!

[2/5] 测试激活函数...
  ✓ ReLU
  ✓ LeakyReLU
  ✓ Sigmoid
  ✓ Tanh
  ✓ Softmax
  ✓ GELU
  ✓ ELU
  ✓ Swish
  所有激活函数测试通过!

[3/5] 测试损失函数...
  ✓ CrossEntropy
  ✓ CrossEntropy (one-hot)
  ✓ MSE
  ✓ BinaryCrossEntropy
  ✓ Huber
  所有损失函数测试通过!

[4/5] 测试优化器...
  ✓ SGD
  ✓ Momentum
  ✓ Adam
  ✓ RMSprop
  ✓ AdaGrad
  ✓ AdaDelta
  所有优化器测试通过!

[5/5] 端到端集成测试 (XOR)...
  XOR 准确率: 100%
  集成测试通过!

=======================================================
  ✅ 全部测试通过!
=======================================================
```

---

## 开发过程中遇到的问题记录

> 在构建这个项目的过程中遇到了若干 bug，以下逐一记录，供参考。

### Bug 1: Conv2D 反向传播矩阵维度错误

**现象:**
```
ValueError: shapes (128,4) and (9,4) not aligned: 4 (dim 1) != 9 (dim 0)
```

**原因:** `Conv2D.backward()` 中计算 `dcol` 时使用了错误的矩阵转置。`dout_reshaped` 的形状是 `(N*out_h*out_w, out_channels)`，而滤波器展平后的形状是 `(out_channels, kH*kW*C)`，两者不能直接用 `.T` 做矩阵乘法。

**修复:**
```python
# 错误写法
dcol = np.dot(dout_reshaped, W_col.T)   # W_col.T 形状不对

# 正确写法
dcol = np.dot(dout_reshaped, self.W.reshape(self.out_channels, -1))
```

**教训:** im2col 方式的卷积反向传播中，`dcol = dout @ W_flat`，其中 `W_flat` 是滤波器的二维展平形式 `(out_channels, in_ch*kH*kW)`。

---

### Bug 2: Conv2D padding 导致 col2im reshape 失败

**现象:**
```
ValueError: cannot reshape array of size 1152 into shape (2,6,6,1,3,3)
```

**原因:** 当卷积层设置了 `padding > 0` 时，`forward()` 中对输入做了零填充（形状变大），但 `_col2im()` 在反向传播时使用的是原始未填充的形状 `self.original_shape`，导致 reshape 维度不匹配。

**修复:**
```python
# forward 中保存填充后的形状
self.padded_shape = x_padded.shape  # 新增

# backward 中使用填充后形状
dx = self._col2im(dcol, self.padded_shape, ...)  # 改为 padded_shape
```

**教训:** 有 padding 的卷积层，forward 要同时保存原始形状和填充形状，backward 用填充形状做 col2im，最后再裁剪掉 padding 部分。

---

### Bug 3: 激活函数不接受 training 参数

**现象:**
```
TypeError: Tanh.forward() got an unexpected keyword argument 'training'
```

**原因:** `Sequential._forward()` 对每一层统一调用 `layer.forward(x, training=training)`，但激活函数的 `forward(x)` 只接受一个参数，没有 `**kwargs`。

**修复:**
```python
class Activation:
    def forward(self, x, **kwargs):   # 加上 **kwargs 兼容 training 参数
        raise NotImplementedError
```

**教训:** 在统一的模型循环中调用各组件时，要确保所有组件接口兼容。激活函数不需要 training 参数，但接收并忽略它是安全的做法。

---

### Bug 4: 优化器对无权重层报错

**现象:**
```
AttributeError: 'Tanh' object has no attribute 'W'
```

**原因:** `optimizer.update(layer)` 在检查 `layer.dW is not None` 之前就尝试访问 `layer.W` 来初始化缓存字典。当层是 Tanh/ReLU 等激活函数时，它们没有 `W` 属性。

**修复:**
```python
class Optimizer:
    @staticmethod
    def _has_weights(layer):
        return hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None

    def update(self, layer):
        if not self._has_weights(layer):  # 提前返回
            return
        # ... 后续逻辑
```

**教训:** 优化器的 `update()` 方法会被模型中的每一层调用（包括无参数的激活函数），必须在入口处做好守卫检查。

---

### Bug 5: XOR 示例中 y[i] 是数组而非标量

**现象:**
```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

**原因:** XOR 的标签 `y` 被 reshape 为 `(4, 1)` 二维数组，`y[i]` 返回的是 shape `(1,)` 的数组而不是标量，无法直接传给 `int()`。

**修复:**
```python
# 错误
int(y[i])  # y[i] 是 array([0])

# 正确
int(y[i][0])  # 取出标量
```

**教训:** NumPy 数组的索引行为：`(N,1)` 形状的数组取第 i 行仍然是 1D 数组，需要额外 `[0]` 取标量。

---

### Bug 6: Matplotlib API 名称拼写错误

**现象:**
```
AttributeError: 'Axes' object has no attribute 'set_value'
AttributeError: 'Axes' object has no attribute 'set_loss'
```

**原因:** Matplotlib Axes 的正确方法名是 `set_ylabel()`，误写成了 `set_value()` 和 `set_loss()`。

**修复:**
```python
axes[0].set_ylabel('Value')     # 不是 set_value()
axes[1].set_ylabel('Loss (MSE)') # 不是 set_loss()
```

**教训:** Matplotlib API 容易混淆，`set_xlabel` / `set_ylabel` / `set_title` 是标准命名模式。

---

### Bug 7: 缺少依赖包

**现象:**
```
ModuleNotFoundError: No module named 'numpy'
ImportError: fetch_openml requires pandas.
```

**原因:**
- 初始环境未安装 numpy/sklearn/matplotlib
- sklearn 的 `fetch_openml` 在较新版本中依赖 pandas

**修复:**
```bash
pip install numpy scikit-learn matplotlib pandas
```

**教训:** `requirements.txt` 应列出所有直接和间接依赖；sklearn 的数据加载功能在不同版本间有依赖变化。

---

## API 参考

### 层 (Layers)

| 类 | 说明 | 关键参数 |
|---|---|---|
| `Dense(in, out)` | 全连接层 | `weight_init`: 'he'/'xavier'/'normal' |
| `Conv2D(in_ch, out_ch, kernel)` | 卷积层 | `stride`, `padding` |
| `MaxPool2D(size)` | 最大池化 | `stride`(默认=size) |
| `Flatten()` | 展平 | 无 |
| `Dropout(rate)` | 随机失活 | `rate`: 失活概率 |
| `BatchNorm(features)` | 批归一化 | `momentum`, `epsilon` |

### 激活函数 (Activations)

| 类 | 公式 | 适用场景 |
|---|---|---|
| `ReLU` | max(0,x) | 通用隐藏层 |
| `LeakyReLU(a)` | x if x>0 else ax | 解决神经元死亡 |
| `Sigmoid` | 1/(1+e^-x) | 二分类输出 |
| `Tanh` | tanh(x) | 零中心化隐藏层 |
| `Softmax` | e^xi / sum(e^xj) | 多分类输出 |
| `GELU` | x * Phi(x) | Transformer/BERT |
| `ELU(a)` | x if x>0 else a(e^x-1) | 负区间非零均值 |
| `Swish` | x * sigmoid(x) | 深层网络 |

### 损失函数 (Losses)

| 类 | 公式 | 任务类型 |
|---|---|---|
| `MSE` | mean((y-y')^2) | 回归 |
| `CrossEntropy` | -mean(log(p)) | 多分类 |
| `BinaryCrossEntropy` | -mean(ylog(p)+(1-y)log(1-p)) | 二分类 |
| `Huber(delta)` | 平方+绝对值混合 | 鲁棒回归 |
| `Hinge(margin)` | max(0, m-y*y') | SVM风格分类 |

### 优化器 (Optimizers)

| 类 | 特点 | 推荐场景 |
|---|---|---|
| `SGD(lr)` | 最基础 | 简单任务 |
| `Momentum(lr, mom)` | 动量加速 | 一般场景 |
| `Adam(lr)` | 自适应+动量 | **默认首选** |
| `RMSprop(lr)` | 自适应学习率 | RNN |
| `AdaGrad(lr)` | 稀疏数据 | NLP |
| `AdaDelta(rho)` | 无需设学习率 | 自动调参 |

### 学习率调度 (LRScheduler)

```python
scheduler = LRScheduler(optimizer,
    mode='step',       # 'step' / 'exponential' / 'cosine'
    step_size=10,      # 阶梯衰减的间隔
    gamma=0.1,         # 衰减因子
    max_epochs=30,     # 余弦退火的总轮数
    min_lr=1e-6)       # 最小学习率下限
```

## 参考 & 致谢

本项目的设计思路参考了以下经典资源：

- **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)** by Michael Nielsen — 神经网络入门圣经
- **[Implementing a Neural Network from Scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)** by Denny Britz — 清晰的代码实现
- **[cs231n](https://cs231n.github.io/)** Stanford — 卷积神经网络课程笔记

## License

MIT License
