"""
第一章：神经网络基础 - 练习题
完成这些练习来巩固你的理解

说明：
1. 每个 TODO 部分需要你填写代码
2. 运行代码检查你的答案
3. 如果卡住了，可以参考 examples/01_simple_neural_network.py
"""

import numpy as np


# ============ 练习1: 实现激活函数 ============

def sigmoid(z):
    """
    TODO: 实现 Sigmoid 激活函数
    
    公式: f(z) = 1 / (1 + e^(-z))
    
    提示: 使用 np.exp()
    
    参数:
        z: 输入值
    
    返回:
        sigmoid(z)
    """
    # 你的代码在这里
    pass


def relu(z):
    """
    TODO: 实现 ReLU 激活函数
    
    公式: f(z) = max(0, z)
    
    提示: 使用 np.maximum()
    
    参数:
        z: 输入值
    
    返回:
        relu(z)
    """
    # 你的代码在这里
    pass


def sigmoid_derivative(a):
    """
    TODO: 实现 Sigmoid 导数
    
    公式: f'(z) = sigmoid(z) * (1 - sigmoid(z))
    
    注意: 这里假设输入 a = sigmoid(z)
    
    参数:
        a: sigmoid激活值
    
    返回:
        sigmoid的导数
    """
    # 你的代码在这里
    pass


def relu_derivative(z):
    """
    TODO: 实现 ReLU 导数
    
    公式: 
        f'(z) = 1, if z > 0
        f'(z) = 0, if z <= 0
    
    提示: 使用布尔运算
    
    参数:
        z: 输入值
    
    返回:
        relu的导数
    """
    # 你的代码在这里
    pass


# ============ 练习2: 实现前向传播 ============

def forward_propagation(X, W1, b1, W2, b2):
    """
    TODO: 实现前向传播
    
    步骤:
    1. 计算隐藏层: Z1 = W1·X + b1, A1 = relu(Z1)
    2. 计算输出层: Z2 = W2·A1 + b2, A2 = sigmoid(Z2)
    
    参数:
        X: 输入数据 (n_features, m_samples)
        W1: 第一层权重 (n_hidden, n_features)
        b1: 第一层偏置 (n_hidden, 1)
        W2: 第二层权重 (n_output, n_hidden)
        b2: 第二层偏置 (n_output, 1)
    
    返回:
        A2: 输出层激活值
        cache: 包含 Z1, A1, Z2, A2 的字典
    """
    # 你的代码在这里
    pass


# ============ 练习3: 实现损失函数 ============

def compute_loss(Y, Y_pred):
    """
    TODO: 实现交叉熵损失函数
    
    公式: L = -(1/m) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
    提示:
    1. 使用 np.log() 计算对数
    2. 注意防止 log(0)，使用 np.clip() 限制范围
    
    参数:
        Y: 真实标签 (n_output, m_samples)
        Y_pred: 预测值 (n_output, m_samples)
    
    返回:
        loss: 损失值
    """
    # 你的代码在这里
    pass


# ============ 练习4: 实现反向传播 ============

def backward_propagation(X, Y, cache, W1, W2):
    """
    TODO: 实现反向传播
    
    步骤:
    1. 输出层梯度:
       dZ2 = A2 - Y
       dW2 = (1/m) * dZ2·A1^T
       db2 = (1/m) * ΣdZ2
    
    2. 隐藏层梯度:
       dZ1 = W2^T·dZ2 * relu'(Z1)
       dW1 = (1/m) * dZ1·X^T
       db1 = (1/m) * ΣdZ1
    
    参数:
        X: 输入数据
        Y: 真实标签
        cache: 前向传播缓存的值 (Z1, A1, Z2, A2)
        W1, W2: 权重矩阵
    
    返回:
        gradients: 包含 dW1, db1, dW2, db2 的字典
    """
    # 你的代码在这里
    pass


# ============ 练习5: 实现参数更新 ============

def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    """
    TODO: 实现参数更新
    
    公式: θ = θ - α * ∇θ
    
    参数:
        W1, b1, W2, b2: 当前参数
        gradients: 梯度字典
        learning_rate: 学习率
    
    返回:
        W1, b1, W2, b2: 更新后的参数
    """
    # 你的代码在这里
    pass


# ============ 测试函数 ============

def test_activation_functions():
    """测试激活函数"""
    print("=" * 60)
    print("测试激活函数")
    print("=" * 60)
    
    z = np.array([-2, -1, 0, 1, 2])
    
    print("\n输入 z:", z)
    
    # 测试 sigmoid
    try:
        sig = sigmoid(z)
        print("\nSigmoid(z):", sig)
        print("期望值: [0.119  0.269  0.5    0.731  0.881]")
        assert np.allclose(sig, [0.119, 0.269, 0.5, 0.731, 0.881], atol=0.01)
        print("✓ Sigmoid 正确!")
    except Exception as e:
        print("✗ Sigmoid 错误:", e)
    
    # 测试 relu
    try:
        r = relu(z)
        print("\nReLU(z):", r)
        print("期望值: [0 0 0 1 2]")
        assert np.array_equal(r, [0, 0, 0, 1, 2])
        print("✓ ReLU 正确!")
    except Exception as e:
        print("✗ ReLU 错误:", e)
    
    # 测试 sigmoid 导数
    try:
        a = sigmoid(z)
        sig_deriv = sigmoid_derivative(a)
        print("\nSigmoid 导数:", sig_deriv)
        print("期望值: [0.105  0.197  0.25   0.197  0.105]")
        assert np.allclose(sig_deriv, [0.105, 0.197, 0.25, 0.197, 0.105], atol=0.01)
        print("✓ Sigmoid 导数正确!")
    except Exception as e:
        print("✗ Sigmoid 导数错误:", e)
    
    # 测试 relu 导数
    try:
        relu_deriv = relu_derivative(z)
        print("\nReLU 导数:", relu_deriv)
        print("期望值: [0 0 0 1 1]")
        assert np.array_equal(relu_deriv, [0, 0, 0, 1, 1])
        print("✓ ReLU 导数正确!")
    except Exception as e:
        print("✗ ReLU 导数错误:", e)
    
    print()


def test_forward_propagation():
    """测试前向传播"""
    print("=" * 60)
    print("测试前向传播")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(2, 3)  # 2个特征，3个样本
    W1 = np.random.randn(4, 2)  # 4个隐藏神经元
    b1 = np.zeros((4, 1))
    W2 = np.random.randn(1, 4)
    b2 = np.zeros((1, 1))
    
    try:
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        print("\n输入 X 形状:", X.shape)
        print("输出 A2 形状:", A2.shape)
        print("输出 A2:", A2)
        
        assert A2.shape == (1, 3)
        assert 'Z1' in cache and 'A1' in cache and 'Z2' in cache and 'A2' in cache
        print("✓ 前向传播正确!")
    except Exception as e:
        print("✗ 前向传播错误:", e)
    
    print()


def test_loss_function():
    """测试损失函数"""
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)
    
    Y = np.array([[1, 0, 1]])
    Y_pred = np.array([[0.9, 0.1, 0.8]])
    
    try:
        loss = compute_loss(Y, Y_pred)
        print("\n真实标签 Y:", Y)
        print("预测值 Y_pred:", Y_pred)
        print("损失值:", loss)
        print("期望值: 约 0.144")
        
        assert 0.1 < loss < 0.2
        print("✓ 损失函数正确!")
    except Exception as e:
        print("✗ 损失函数错误:", e)
    
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试")
    print("=" * 60 + "\n")
    
    test_activation_functions()
    test_forward_propagation()
    test_loss_function()
    
    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


# ============ 挑战题 ============

def challenge():
    """
    挑战题: 实现一个完整的神经网络并训练
    
    任务:
    1. 使用上面的函数构建一个完整的神经网络
    2. 在 XOR 问题上训练
    3. 达到 95% 以上的准确率
    
    XOR 问题:
        [0, 0] -> 0
        [0, 1] -> 1
        [1, 0] -> 1
        [1, 1] -> 0
    """
    print("\n" + "=" * 60)
    print("挑战题: XOR 问题")
    print("=" * 60)
    
    # XOR 数据集
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    
    print("\nXOR 数据集:")
    print("X =", X)
    print("Y =", Y)
    
    # TODO: 实现你的神经网络
    # 提示:
    # 1. 初始化参数
    # 2. 训练循环
    # 3. 预测
    
    print("\n完成挑战题后，运行 predict() 检查准确率")


if __name__ == "__main__":
    # 运行测试
    run_all_tests()
    
    # 挑战题
    challenge()
