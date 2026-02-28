"""
第一章：神经网络基础 - 完整示例
从零实现一个简单的神经网络

运行方式：python 01_simple_neural_network.py
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuralNetwork:
    """简单的神经网络实现"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络
        
        参数:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
        
        # 存储损失历史
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Sigmoid导数"""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据 (input_size, m_samples)
        
        返回:
            A2: 输出层激活值
            cache: 缓存的中间值
        """
        # 隐藏层
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        
        # 输出层
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        
        return A2, cache
    
    def compute_loss(self, Y, Y_pred):
        """
        计算交叉熵损失
        
        参数:
            Y: 真实标签 (output_size, m_samples)
            Y_pred: 预测值 (output_size, m_samples)
        
        返回:
            loss: 损失值
        """
        m = Y.shape[1]
        
        # 防止log(0)
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        
        # 交叉熵损失
        loss = -np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred)) / m
        
        return loss
    
    def backward(self, X, Y, cache):
        """
        反向传播
        
        参数:
            X: 输入数据
            Y: 真实标签
            cache: 前向传播缓存的值
        
        返回:
            gradients: 梯度字典
        """
        m = X.shape[1]
        
        A1 = cache["A1"]
        A2 = cache["A2"]
        Z1 = cache["Z1"]
        
        # 输出层梯度
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        
        # 隐藏层梯度
        dZ1 = np.dot(self.W2.T, dZ2) * self.relu_derivative(Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
        gradients = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2
        }
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        更新参数
        
        参数:
            gradients: 梯度字典
            learning_rate: 学习率
        """
        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]
        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]
    
    def train(self, X, Y, epochs, learning_rate, print_loss=True):
        """
        训练神经网络
        
        参数:
            X: 训练数据
            Y: 训练标签
            epochs: 训练轮数
            learning_rate: 学习率
            print_loss: 是否打印损失
        """
        for epoch in range(epochs):
            # 前向传播
            Y_pred, cache = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(Y, Y_pred)
            self.loss_history.append(loss)
            
            # 反向传播
            gradients = self.backward(X, Y, cache)
            
            # 更新参数
            self.update_parameters(gradients, learning_rate)
            
            # 打印损失
            if print_loss and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            predictions: 预测结果
        """
        Y_pred, _ = self.forward(X)
        predictions = (Y_pred > 0.5).astype(int)
        return predictions
    
    def plot_loss(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        print("损失曲线已保存: training_loss.png")
        plt.close()


def create_dataset():
    """创建一个简单的二分类数据集"""
    np.random.seed(42)
    
    # 生成两类数据
    m = 400  # 样本数
    
    # 类别0：圆形分布
    theta = np.linspace(0, 2*np.pi, m//2)
    r = 2 + np.random.randn(m//2) * 0.3
    X0 = np.array([r * np.cos(theta), r * np.sin(theta)])
    
    # 类别1：中心分布
    X1 = np.random.randn(2, m//2) * 0.5
    
    # 合并数据
    X = np.hstack([X0, X1])
    Y = np.hstack([np.zeros((1, m//2)), np.ones((1, m//2))])
    
    # 打乱数据
    permutation = np.random.permutation(m)
    X = X[:, permutation]
    Y = Y[:, permutation]
    
    return X, Y


def visualize_dataset(X, Y):
    """可视化数据集"""
    plt.figure(figsize=(10, 8))
    plt.scatter(X[0, Y[0, :]==0], X[1, Y[0, :]==0], c='red', label='Class 0', alpha=0.6)
    plt.scatter(X[0, Y[0, :]==1], X[1, Y[0, :]==1], c='blue', label='Class 1', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dataset Visualization')
    plt.legend()
    plt.grid(True)
    plt.savefig('dataset.png', dpi=300, bbox_inches='tight')
    print("数据集可视化已保存: dataset.png")
    plt.close()


def visualize_decision_boundary(model, X, Y):
    """可视化决策边界"""
    # 创建网格
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, Y[0, :]==0], X[1, Y[0, :]==0], c='red', label='Class 0', edgecolors='black')
    plt.scatter(X[0, Y[0, :]==1], X[1, Y[0, :]==1], c='blue', label='Class 1', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
    print("决策边界已保存: decision_boundary.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("神经网络基础示例")
    print("=" * 60)
    print()
    
    # 1. 创建数据集
    print("步骤1: 创建数据集...")
    X, Y = create_dataset()
    print(f"数据集形状: X={X.shape}, Y={Y.shape}")
    visualize_dataset(X, Y)
    print()
    
    # 2. 创建神经网络
    print("步骤2: 创建神经网络...")
    input_size = X.shape[0]  # 2个特征
    hidden_size = 10         # 隐藏层10个神经元
    output_size = 1          # 二分类
    
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    print(f"网络结构: {input_size} -> {hidden_size} -> {output_size}")
    print()
    
    # 3. 训练神经网络
    print("步骤3: 训练神经网络...")
    print("-" * 60)
    model.train(X, Y, epochs=1000, learning_rate=0.5, print_loss=True)
    print("-" * 60)
    print()
    
    # 4. 评估模型
    print("步骤4: 评估模型...")
    predictions = model.predict(X)
    accuracy = np.mean(predictions == Y) * 100
    print(f"训练集准确率: {accuracy:.2f}%")
    print()
    
    # 5. 可视化结果
    print("步骤5: 可视化结果...")
    model.plot_loss()
    visualize_decision_boundary(model, X, Y)
    print()
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. dataset.png - 数据集可视化")
    print("  2. training_loss.png - 训练损失曲线")
    print("  3. decision_boundary.png - 决策边界")


if __name__ == "__main__":
    main()
