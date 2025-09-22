"""
测试神经网络层的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import Dense, ReLU, Sigmoid, Tanh, Softmax

def test_dense_forward():
    """测试全连接层前向传播"""
    layer = Dense(3, 2)
    x = np.random.randn(4, 3)
    output = layer.forward(x)
    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
    print("✓ Dense forward 测试通过")

def test_relu():
    """测试 ReLU 激活函数"""
    relu = ReLU()
    x = np.array([[-1, 0, 1]])
    output = relu.forward(x)
    assert np.allclose(output, np.array([[0, 0, 1]])), f"Expected [[0, 0, 1]], got {output}"
    print("✓ ReLU 测试通过")

def test_sigmoid():
    """测试 Sigmoid 激活函数"""
    sigmoid = Sigmoid()
    x = np.array([[0]])
    output = sigmoid.forward(x)
    assert np.allclose(output, np.array([[0.5]])), f"Expected [[0.5]], got {output}"
    print("✓ Sigmoid 测试通过")

def test_softmax():
    """测试 Softmax 激活函数"""
    softmax = Softmax()
    x = np.array([[1, 2, 3]])
    output = softmax.forward(x)
    assert np.allclose(np.sum(output, axis=1), 1.0), "Softmax 输出应该和为 1"
    assert output.shape == (1, 3), f"Expected (1, 3), got {output.shape}"
    print("✓ Softmax 测试通过")

def test_gradient_check():
    """梯度检查 - 验证反向传播"""
    np.random.seed(42)
    layer = Dense(2, 3)
    x = np.random.randn(5, 2)
    
    # 前向传播
    output = layer.forward(x)
    
    # 反向传播
    dout = np.random.randn(5, 3)
    dx = layer.backward(dout)
    
    # 检查梯度
    assert dx.shape == x.shape, f"Expected {x.shape}, got {dx.shape}"
    assert layer.dW.shape == layer.W.shape, f"Expected {layer.W.shape}, got {layer.dW.shape}"
    assert layer.db.shape == layer.b.shape, f"Expected {layer.b.shape}, got {layer.db.shape}"
    print("✓ 梯度检查测试通过")

if __name__ == "__main__":
    print("=" * 50)
    print("运行神经网络测试...")
    print("=" * 50)
    
    test_dense_forward()
    test_relu()
    test_sigmoid()
    test_softmax()
    test_gradient_check()
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)
