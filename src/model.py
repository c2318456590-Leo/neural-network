"""
神经网络模型模块

提供 Sequential 顺序模型，支持完整的训练流程：
- 前向传播 → 计算损失 → 反向传播 → 参数更新
- 早停机制 (Early Stopping)：验证损失不再下降时停止训练
- 训练历史记录：跟踪 loss 和 accuracy 变化曲线
- 模型保存/加载：持久化训练好的权重
"""

import pickle
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layers import Layer
from src.optimizers import Optimizer, LRScheduler


# ============================================================
# 早停机制 (Early Stopping)
# ============================================================
class EarlyStopping:
    """
    早停回调：当验证损失连续 patience 个 epoch 不再改善时停止训练

    参数:
        patience: 容忍多少个 epoch 无改善
        min_delta: 视为"改善"的最小变化量
        restore_best_weights: 是否恢复到最佳权重的状态
    """
    def __init__(self, patience=5, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, val_loss, model, epoch):
        """
        根据当前验证损失判断是否应该早停
        返回 True 表示应停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.get_weights()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n  [EarlyStopping] Epoch {epoch}: "
                      f"val_loss 未改善 {self.patience} 轮，触发早停 "
                      f"(best at epoch {self.best_epoch})")
                if self.restore_best_weights and self.best_weights:
                    model.set_weights(self.best_weights)
                    print(f"  [EarlyStopping] 已恢复最佳权重")
                self.should_stop = True
                return True
        return False


# ============================================================
# 训练历史记录 (Training History)
# ============================================================
class History:
    """记录训练过程中的各项指标"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

    def record(self, train_loss, val_loss=None,
               train_acc=None, val_acc=None, lr=None, epoch_time=0):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)


# ============================================================
# Sequential 顺序模型
# ============================================================
class Sequential:
    """
    顺序神经网络模型：按顺序堆叠各层

    用法示例:
        model = Sequential()
        model.add(Dense(784, 256))
        model.add(ReLU())
        model.add(Dense(256, 10))
        model.add(Softmax())
        model.compile(optimizer=Adam(), loss_fn=CrossEntropy())
        history = model.fit(X_train, y_train, epochs=20, batch_size=32)
    """

    def __init__(self):
        self.layers = []
        self.compiled = False
        self.optimizer = None
        self.loss_fn = None
        self.history = History()

    def add(self, layer):
        """向模型中添加一层"""
        self.layers.append(layer)
        return self

    def compile(self, optimizer=None, loss_fn=None):
        """
        编译模型：设置优化器和损失函数
        可在 fit 时指定，也可提前 compile
        """
        if optimizer:
            self.optimizer = optimizer
        if loss_fn:
            self.loss_fn = loss_fn
        self.compiled = True

    def _forward(self, x, training=True):
        """前向传播：依次通过每一层"""
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def _backward(self, dout):
        """反向传播：从输出层向输入层逐层传递梯度"""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def _update_params(self):
        """使用优化器更新所有可训练层的参数"""
        for layer in self.layers:
            self.optimizer.update(layer)

    def fit(self, X, y, epochs=100, batch_size=32,
            validation_data=None, optimizer=None, loss_fn=None,
            verbose=True, early_stopping=None, lr_scheduler=None):
        """
        训练模型的完整流程

        参数:
            X: 训练数据 (N, features)
            y: 标签
            epochs: 训练轮数
            batch_size: 批次大小
            validation_data: (X_val, y_val) 验证集
            optimizer: 优化器实例
            loss_fn: 损失函数实例
            verbose: 是否打印训练日志
            early_stopping: EarlyStopping 回调实例
            lr_scheduler: LRScheduler 学习率调度器实例

        返回:
            History 对象，包含训练历史数据
        """
        # 编译
        if optimizer:
            self.optimizer = optimizer
        if loss_fn:
            self.loss_fn = loss_fn

        if self.optimizer is None:
            from src.optimizers import SGD
            self.optimizer = SGD()
        if self.loss_fn is None:
            from src.losses import CrossEntropy
            self.loss_fn = CrossEntropy()

        self.history = History()
        n_samples = X.shape[0]

        if early_stopping:
            early_stopping.should_stop = False

        print(f"开始训练 | 样本数: {n_samples}, 批次: {batch_size}, "
              f"轮数: {epochs}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            correct = 0
            total = 0
            n_batches = 0

            # Mini-batch 训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # 前向传播
                y_pred = self._forward(X_batch, training=True)

                # 计算损失
                loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss

                # 计算准确率
                predictions = self._predict_classes(y_pred, y_batch)
                correct += int(predictions)
                total += len(y_batch)
                n_batches += 1

                # 反向传播
                dout = self.loss_fn.backward()
                self._backward(dout)

                # 参数更新
                self._update_params()

            # 本 epoch 统计
            avg_loss = epoch_loss / max(n_batches, 1)
            train_acc = correct / max(total, 1)
            epoch_time = time.time() - epoch_start
            current_lr = getattr(self.optimizer, 'lr', None)

            # 验证集评估
            val_loss = None
            val_acc = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size=batch_size)

            # 记录历史
            self.history.record(avg_loss, val_loss, train_acc, val_acc,
                                current_lr, epoch_time)

            # 学习率调度
            if lr_scheduler:
                lr_scheduler.step()

            # 打印日志
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1 or epoch == epochs):
                log_msg = (f"Epoch {epoch:>3d}/{epochs} | "
                           f"loss: {avg_loss:.4f} | acc: {train_acc:.4f}")
                if val_loss is not None:
                    log_msg += f" | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
                if current_lr is not None:
                    log_msg += f" | lr: {current_lr:.6f}"
                log_msg += f" | {epoch_time:.1f}s"
                print(log_msg)

            # 早停检查
            if early_stopping and val_loss is not None:
                if early_stopping(val_loss, self, epoch):
                    break

        print("=" * 60)
        print("训练完成!")
        return self.history

    def predict(self, X):
        """预测: 返回类别索引"""
        output = self._forward(X, training=False)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """预测概率: 返回原始输出（Softmax 后的概率）"""
        return self._forward(X, training=False)

    def evaluate(self, X, y, batch_size=None):
        """
        评估模型性能
        返回: (平均损失, 准确率)
        """
        if batch_size is None:
            batch_size = len(X)

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            y_pred = self._forward(X_batch, training=False)
            loss = self.loss_fn.forward(y_pred, y_batch)
            total_loss += loss

            predictions = self._predict_classes(y_pred, y_batch)
            correct += int(predictions)
            total += len(y_batch)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _predict_classes(self, y_pred, y_true):
        """根据输出和标签格式计算正确的预测数量"""
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = y_true.flatten().astype(int)
        return int(np.sum(pred_classes == true_classes))

    def summary(self):
        """打印模型结构摘要"""
        total_params = 0
        print("\n" + "=" * 60)
        print(" Model Summary")
        print("=" * 60)
        print(f"{'Layer':<25s} {'Output Shape':<22s} {'Params':>8s}")
        print("-" * 60)

        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            params = 0
            if hasattr(layer, 'W'):
                params += layer.W.size
            if hasattr(layer, 'b'):
                params += layer.b.size
            if hasattr(layer, 'gamma'):
                params += layer.gamma.size + layer.beta.size
            total_params += params
            print(f"{i+1}. {name:<23s} {'--':>20s} {params:>8,}")

        print("-" * 60)
        print(f"Total Parameters: {total_params:,}")
        print("=" * 60 + "\n")

    def get_weights(self):
        """获取所有层的权重"""
        weights = []
        for layer in self.layers:
            w = {}
            if hasattr(layer, 'W'):
                w['W'] = layer.W.copy()
            if hasattr(layer, 'b'):
                w['b'] = layer.b.copy()
            if hasattr(layer, 'gamma'):
                w['gamma'] = layer.gamma.copy()
                w['beta'] = layer.beta.copy()
            if hasattr(layer, 'running_mean'):
                w['running_mean'] = layer.running_mean.copy()
                w['running_var'] = layer.running_var.copy()
            weights.append(w)
        return weights

    def set_weights(self, weights):
        """设置所有层的权重"""
        for layer, w in zip(self.layers, weights):
            if 'W' in w:
                layer.W = w['W'].copy()
            if 'b' in w:
                layer.b = w['b'].copy()
            if 'gamma' in w:
                layer.gamma = w['gamma'].copy()
                layer.beta = w['beta'].copy()
            if 'running_mean' in w:
                layer.running_mean = w['running_mean'].copy()
                layer.running_var = w['running_var'].copy()

    def save(self, filepath):
        """保存模型到文件"""
        data = {
            'weights': self.get_weights(),
            'layer_names': [l.__class__.__name__ for l in self.layers],
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"模型已保存至: {filepath}")

    def load(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.set_weights(data['weights'])
        print(f"模型已从 {filepath} 加载")
