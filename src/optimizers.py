"""
优化器模块

负责根据计算出的梯度更新网络参数：

- SGD: 随机梯度下降（基础优化器）
- Momentum: 动量加速的 SGD（加速收敛）
- RMSprop: 自适应学习率（适合 RNN）
- Adam: 动量 + 自适应学习率（最常用，综合表现好）
- AdaGrad: 累积平方梯度（适合稀疏数据）
- AdaDelta: AdaGrad 的改进版（无需手动设置学习率）

每个优化器都实现统一的 update(layer) 接口。
"""

import numpy as np


class Optimizer:
    """所有优化器的基类"""
    def update(self, layer):
        raise NotImplementedError

    def step(self):
        """每步结束后的回调（如学习率衰减），子类可选覆盖"""
        pass

    @staticmethod
    def _has_weights(layer):
        """检查层是否有可训练的权重参数"""
        return hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None


# ============================================================
# SGD (Stochastic Gradient Descent)
# ============================================================
class SGD(Optimizer):
    """
    随机梯度下降: param = param - lr * gradient
    最基础的优化器，但仍然非常有效
    """
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer):
        if not self._has_weights(layer):
            return
        if hasattr(layer, 'W') and layer.dW is not None:
            reg_grad = layer.dW + self.weight_decay * layer.W
            layer.W -= self.lr * reg_grad
            layer.b -= self.lr * layer.db


# ============================================================
# Momentum
# ============================================================
class Momentum(Optimizer):
    """
    动量 SGD: 引入物理中的"惯性"概念
      v_t = momentum * v_{t-1} - lr * grad
      param = param + v_t
    优点：加速收敛、帮助逃离局部最优/鞍点
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layer):
        if not self._has_weights(layer):
            return
        lid = id(layer)
        if lid not in self.velocities:
            self.velocities[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b)
            }

        v = self.velocities[lid]
        if hasattr(layer, 'W') and layer.dW is not None:
            grad_w = layer.dW + self.weight_decay * layer.W
            v['W'] = self.momentum * v['W'] - self.lr * grad_w
            v['b'] = self.momentum * v['b'] - self.lr * layer.db
            layer.W += v['W']
            layer.b += v['b']


# ============================================================
# RMSprop
# ============================================================
class RMSprop(Optimizer):
    """
    RMSprop: 自适应学习率优化器
      s_t = rho * s_{t-1} + (1-rho) * grad^2
      param = param - lr * grad / (sqrt(s_t) + eps)
    特点：为每个参数自适应调整学习率，适合处理非平稳目标（如 RNN）
    """
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8,
                 weight_decay=0.0):
        self.lr = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}

    def update(self, layer):
        if not self._has_weights(layer):
            return
        lid = id(layer)
        if lid not in self.cache:
            self.cache[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b)
            }

        c = self.cache[lid]
        if hasattr(layer, 'W') and layer.dW is not None:
            grad_w = layer.dW + self.weight_decay * layer.W
            c['W'] = self.rho * c['W'] + (1 - self.rho) * (grad_w ** 2)
            c['b'] = self.rho * c['b'] + (1 - self.rho) * (layer.db ** 2)
            layer.W -= self.lr * grad_w / (np.sqrt(c['W']) + self.epsilon)
            layer.b -= self.lr * layer.db / (np.sqrt(c['b']) + self.epsilon)


# ============================================================
# Adam (Adaptive Moment Estimation)
# ============================================================
class Adam(Optimizer):
    """
    Adam 优化器: 结合动量和自适应学习率
      m_t = beta1*m_{t-1} + (1-beta1)*grad     （一阶矩估计）
      v_t = beta2*v_{t-1} + (1-beta2)*grad^2   （二阶矩估计）
      m_hat = m_t / (1-beta1^t)                 （偏差校正）
      v_hat = v_t / (1-beta2^t)
      param = param - lr * m_hat / (sqrt(v_hat) + eps)

    目前深度学习中最流行的优化器之一
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        if not self._has_weights(layer):
            return
        self.t += 1
        lid = id(layer)
        if lid not in self.m:
            self.m[lid] = {'W': np.zeros_like(layer.W),
                           'b': np.zeros_like(layer.b)}
            self.v[lid] = {'W': np.zeros_like(layer.W),
                           'b': np.zeros_like(layer.b)}

        m = self.m[lid]
        v = self.v[lid]
        if hasattr(layer, 'W') and layer.dW is not None:
            grad_w = layer.dW + self.weight_decay * layer.W

            # 一阶矩（动量）
            m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * grad_w
            m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * layer.db

            # 二阶矩（自适应学习率）
            v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * (grad_w ** 2)
            v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * (layer.db ** 2)

            # 偏差校正
            m_hat_w = m['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = m['b'] / (1 - self.beta1 ** self.t)
            v_hat_w = v['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = v['b'] / (1 - self.beta2 ** self.t)

            # 参数更新
            layer.W -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


# ============================================================
# AdaGrad
# ============================================================
class AdaGrad(Optimizer):
    """
    AdaGrad: 自适应梯度算法
      G_t = G_{t-1} + grad^2
      param = param - lr * grad / (sqrt(G_t) + eps)
    特点：频繁更新的参数学习率自动减小，稀疏参数保持较大学习率
    适用于稀疏数据（如自然语言处理）
    缺点：学习率单调递减，后期可能过小
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.G = {}

    def update(self, layer):
        if not self._has_weights(layer):
            return
        lid = id(layer)
        if lid not in self.G:
            self.G[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b)
            }

        G = self.G[lid]
        if hasattr(layer, 'W') and layer.dW is not None:
            grad_w = layer.dW + self.weight_decay * layer.W
            G['W'] += grad_w ** 2
            G['b'] += layer.db ** 2
            layer.W -= self.lr * grad_w / (np.sqrt(G['W']) + self.epsilon)
            layer.b -= self.lr * layer.db / (np.sqrt(G['b']) + self.epsilon)


# ============================================================
# AdaDelta
# ============================================================
class AdaDelta(Optimizer):
    """
    AdaDelta: AdaGrad 的改进版，解决了学习率递减的问题
      E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
      RMS[x]_{t-1} = sqrt(E[deltax^2]_{t-1} + eps)
      deltax_t = - (RMS[x]_{t-1} / sqrt(E[g^2]_t + eps)) * g_t
      E[deltax^2]_t = rho * E[deltax^2]_{t-1} + (1-rho) * deltax_t^2
    优点：无需手动设置学习率！
    """
    def __init__(self, rho=0.95, epsilon=1e-6, weight_decay=0.0):
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.E_g2 = {}  # 梯度平方的期望
        self.E_dx2 = {}  # 参数更新量的期望

    def update(self, layer):
        if not self._has_weights(layer):
            return
        lid = id(layer)
        if lid not in self.E_g2:
            shape_W, shape_b = layer.W.shape, layer.b.shape
            self.E_g2[lid] = {'W': np.zeros(shape_W), 'b': np.zeros(shape_b)}
            self.E_dx2[lid] = {'W': np.zeros(shape_W), 'b': np.zeros(shape_b)}

        Eg = self.E_g2[lid]
        Edx = self.E_dx2[lid]
        if hasattr(layer, 'W') and layer.dW is not None:
            grad_w = layer.dW + self.weight_decay * layer.W

            for param_name, grad, param in [
                ('W', grad_w, layer.W),
                ('b', layer.db, layer.b)
            ]:
                # 累积梯度平方
                Eg[param_name] = (
                    self.rho * Eg[param_name] +
                    (1 - self.rho) * (grad ** 2)
                )

                # 计算参数更新量
                rms_dx = np.sqrt(Edx[param_name] + self.epsilon)
                rms_g = np.sqrt(Eg[param_name] + self.epsilon)
                dx = -rms_dx / rms_g * grad

                # 累积更新量平方
                Edx[param_name] = (
                    self.rho * Edx[param_name] +
                    (1 - self.rho) * (dx ** 2)
                )

                # 更新参数
                param += dx


# ============================================================
# 学习率调度器 (Learning Rate Scheduler)
# ============================================================
class LRScheduler:
    """
    学习率调度器：在训练过程中动态调整学习率

    支持的策略:
      - 'step': 每隔 step_size 个 epoch 将学习率乘以 gamma
      - 'exponential': 每个 epoch 学习率乘以 gamma
      - 'cosine': 余弦退火调度
      - 'constant': 保持不变（默认）
    """
    def __init__(self, optimizer, mode='step',
                 step_size=10, gamma=0.1,
                 max_epochs=None, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.step_size = step_size
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        """在每个 epoch 结束后调用，更新学习率"""
        self.current_epoch += 1
        new_lr = self._get_lr()
        new_lr = max(new_lr, self.min_lr)
        self.optimizer.lr = new_lr

    def _get_lr(self):
        if self.mode == 'step':
            # 阶梯衰减
            n_steps = self.current_epoch // self.step_size
            return self.base_lr * (self.gamma ** n_steps)

        elif self.mode == 'exponential':
            # 指数衰减
            return self.base_lr * (self.gamma ** self.current_epoch)

        elif self.mode == 'cosine':
            # 余弦退火
            if self.max_epochs is None:
                return self.base_lr
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * self.current_epoch / self.max_epochs)
            )

        else:
            return self.base_lr
