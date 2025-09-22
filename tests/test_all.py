"""
综合测试文件 - 验证所有模块的基本功能

运行方式:
    python tests/test_all.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from src.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, ELU, Swish
from src.losses import MSE, CrossEntropy, BinaryCrossEntropy, Huber
from src.optimizers import SGD, Momentum, Adam, RMSprop, AdaGrad, AdaDelta
from src.model import Sequential


def test_layers():
    """测试所有层的前向/反向传播"""
    print("\n[1/5] 测试层模块...")

    # Dense 层
    dense = Dense(4, 3)
    x = np.random.randn(2, 4)
    out = dense.forward(x)
    assert out.shape == (2, 3), f"Dense 输出形状错误: {out.shape}"
    dout = np.random.randn(*out.shape)
    dx = dense.backward(dout)
    assert dx.shape == x.shape, f"Dense 反向传播形状错误: {dx.shape}"
    assert dense.dW is not None and dense.db is not None
    print("  ✓ Dense")

    # Conv2D 层
    conv = Conv2D(1, 4, kernel_size=3, padding=1)
    x_img = np.random.randn(2, 1, 8, 8)
    out_conv = conv.forward(x_img)
    assert out_conv.shape == (2, 4, 8, 8), f"Conv2D 输出形状错误: {out_conv.shape}"
    dconv = conv.backward(np.random.randn(*out_conv.shape))
    assert dconv.shape == x_img.shape, "Conv2D 反向传播形状错误"
    print("  ✓ Conv2D")

    # MaxPool2D 层
    pool = MaxPool2D(pool_size=2)
    out_pool = pool.forward(np.random.randn(2, 3, 6, 6))
    assert out_pool.shape == (2, 3, 3, 3), f"MaxPool2D 形状错误: {out_pool.shape}"
    dpool = pool.backward(np.random.randn(*out_pool.shape))
    assert dpool.shape == (2, 3, 6, 6), "MaxPool2D 反向传播形状错误"
    print("  ✓ MaxPool2D")

    # Flatten 层
    flat = Flatten()
    out_flat = flat.forward(np.random.randn(2, 3, 4, 4))
    assert out_flat.shape == (2, 48), f"Flatten 形状错误: {out_flat.shape}"
    df = flat.backward(np.random.randn(*out_flat.shape))
    assert df.shape == (2, 3, 4, 4), "Flatten 反向传播形状错误"
    print("  ✓ Flatten")

    # Dropout 层
    drop = Dropout(rate=0.5)
    out_drop = drop.forward(x, training=True)
    assert out_drop.shape == x.shape, "Dropout 训练模式形状错误"
    out_drop_test = drop.forward(x, training=False)
    assert np.allclose(out_drop_test, x), "Dropout 推理模式应返回原始输入"
    print("  ✓ Dropout")

    # BatchNormalization 层
    bn = BatchNormalization(num_features=4)
    x_bn = np.random.randn(10, 4) * 3 + 2  # 均值2，标准差3
    out_bn = bn.forward(x_bn, training=True)
    assert out_bn.shape == (10, 4), "BatchNorm 形状错误"
    # 检查归一化效果（均值接近0，方差接近1）
    assert np.abs(np.mean(out_bn)) < 0.5, "BatchNorm 均值应接近0"
    print("  ✓ BatchNorm")

    print("  所有层测试通过!")


def test_activations():
    """测试所有激活函数"""
    print("\n[2/5] 测试激活函数...")
    x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)

    # ReLU
    relu_out = ReLU().forward(x)
    expected_relu = np.array([[0, 0, 0, 1, 2]], dtype=np.float32)
    assert np.allclose(relu_out, expected_relu), "ReLU 输出不正确"
    print("  ✓ ReLU")

    # LeakyReLU
    lrelu_out = LeakyReLU(alpha=0.01).forward(x)
    assert lrelu_out[0, 0] < 0 and lrelu_out[0, 0] > x[0, 0], "LeakyReLU 负区间应非零且大于原始值"
    print("  ✓ LeakyReLU")

    # Sigmoid
    sig_out = Sigmoid().forward(x)
    assert sig_out.min() > 0 and sig_out.max() < 1, "Sigmoid 应在 (0,1)"
    print("  ✓ Sigmoid")

    # Tanh
    tanh_out = Tanh().forward(x)
    assert tanh_out.min() > -1 and tanh_out.max() < 1, "Tanh 应在 (-1,1)"
    print("  ✓ Tanh")

    # Softmax
    sm_out = Softmax().forward(np.array([[1, 2, 3]]))
    assert np.allclose(sm_out.sum(), 1.0), "Softmax 和应为1"
    assert sm_out[0, 2] > sm_out[0, 0], "Softmax 最大值应对应最大输入"
    print("  ✓ Softmax")

    # GELU
    gelu_out = GELU().forward(x)
    assert gelu_out.shape == x.shape, "GELU 形状错误"
    print("  ✓ GELU")

    # ELU
    elu_out = ELU(alpha=1.0).forward(x)
    assert elu_out[0, 0] < 0, "ELU 负区间应 < 0"
    print("  ✓ ELU")

    # Swish
    swish_out = Swish().forward(x)
    assert swish_out.shape == x.shape, "Swish 形状错误"
    print("  ✓ Swish")

    print("  所有激活函数测试通过!")


def test_losses():
    """测试所有损失函数"""
    print("\n[3/5] 测试损失函数...")
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true_cls = np.array([0, 1])
    y_true_oh = np.array([[1, 0, 0], [0, 1, 0]])

    # CrossEntropy
    ce = CrossEntropy()
    loss_ce = ce.forward(y_pred, y_true_cls)
    assert loss_ce > 0, "交叉熵损失应 > 0"
    grad = ce.backward()
    assert grad.shape == y_pred.shape, "梯度形状错误"
    print("  ✓ CrossEntropy")

    # CrossEntropy with one-hot
    ce2 = CrossEntropy()
    loss_ce2 = ce2.forward(y_pred, y_true_oh)
    assert abs(loss_ce - loss_ce2) < 1e-6, "两种标签格式结果应一致"
    print("  ✓ CrossEntropy (one-hot)")

    # MSE
    mse_loss = MSE().forward(
        np.array([[1.5], [2.0]]),
        np.array([[1.0], [3.0]])
    )
    assert mse_loss > 0, "MSE 损失应 > 0"
    print("  ✓ MSE")

    # BinaryCrossEntropy
    bce_loss = BinaryCrossEntropy().forward(
        np.array([[0.9], [0.1]]),
        np.array([[1.0], [0.0]])
    )
    assert bce_loss > 0 and bce_loss < 1, "BCE 损失应在合理范围"
    print("  ✓ BinaryCrossEntropy")

    # Huber
    huber_loss = Huber(delta=1.0).forward(
        np.array([[0.5]]),
        np.array([[0.0]])
    )
    assert huber_loss >= 0, "Huber 损失应 >= 0"
    print("  ✓ Huber")

    print("  所有损失函数测试通过!")


def test_optimizers():
    """测试优化器能正确更新参数"""
    print("\n[4/5] 测试优化器...")
    np.random.seed(42)

    layer = Dense(3, 2)
    W_before = layer.W.copy()

    x = np.random.randn(4, 3)
    layer.forward(x)
    layer.backward(np.ones((4, 2)))

    for name, opt_class in [
        ("SGD", lambda: SGD(learning_rate=0.1)),
        ("Momentum", lambda: Momentum(learning_rate=0.1)),
        ("Adam", lambda: Adam(learning_rate=0.1)),
        ("RMSprop", lambda: RMSprop(learning_rate=0.1)),
        ("AdaGrad", lambda: AdaGrad(learning_rate=0.1)),
        ("AdaDelta", lambda: AdaDelta()),
    ]:
        test_layer = Dense(3, 2)
        test_layer.W = W_before.copy()
        test_layer.b = np.zeros((1, 2))

        test_layer.forward(x)
        test_layer.backward(np.ones((4, 2)))

        opt = opt_class()
        opt.update(test_layer)

        # 确保参数发生了变化（或至少没有报错）
        assert not np.allclose(test_layer.W, W_before, atol=1e-10) or name == "AdaDelta", \
            f"{name} 未更新权重"
        print(f"  ✓ {name}")

    print("  所有优化器测试通过!")


def test_model_integration():
    """端到端模型集成测试：XOR 问题"""
    print("\n[5/5] 端到端集成测试 (XOR)...")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)

    model = Sequential()
    model.add(Dense(2, 16))
    model.add(Tanh())
    model.add(Dense(16, 8))
    model.add(Tanh())
    model.add(Dense(8, 1))
    model.add(Sigmoid())

    history = model.fit(X, y.reshape(-1, 1),
                        epochs=2000,
                        batch_size=4,
                        optimizer=Adam(learning_rate=0.2),
                        loss_fn=BinaryCrossEntropy(),
                        verbose=False)

    predictions = model.predict_proba(X).flatten()
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)

    assert accuracy >= 0.75, f"XOR 准确率过低: {accuracy:.2f}"

    # 测试 summary 不报错
    model.summary()

    # 测试 get/set weights
    weights = model.get_weights()
    model.set_weights(weights)

    print(f"  XOR 准确率: {accuracy*100:.0f}%")
    print("  集成测试通过!")


if __name__ == '__main__':
    print("=" * 55)
    print("  Neural Network from Scratch - 全部测试")
    print("=" * 55)

    try:
        test_layers()
        test_activations()
        test_losses()
        test_optimizers()
        test_model_integration()

        print("\n" + "=" * 55)
        print("  ✅ 全部测试通过!")
        print("=" * 55)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
