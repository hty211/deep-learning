import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def test_derivatives():
    print("=" * 50)
    print("导数练习")
    print("=" * 50)
    
    print("\n练习1: 基本导数")
    x = 2.0
    
    f1 = lambda x: x**3
    d1 = numerical_derivative(f1, x)
    print(f"d/dx(x³) 在x={x}: 数值={d1:.6f}, 理论={3*x**2:.6f}")
    
    f2 = lambda x: np.exp(x)
    d2 = numerical_derivative(f2, x)
    print(f"d/dx(e^x) 在x={x}: 数值={d2:.6f}, 理论={np.exp(x):.6f}")
    
    f3 = lambda x: np.log(x)
    d3 = numerical_derivative(f3, x)
    print(f"d/dx(ln(x)) 在x={x}: 数值={d3:.6f}, 理论={1/x:.6f}")
    
    f4 = lambda x: np.sin(x)
    d4 = numerical_derivative(f4, x)
    print(f"d/dx(sin(x)) 在x={x}: 数值={d4:.6f}, 理论={np.cos(x):.6f}")
    
    print("\n练习2: 链式法则")
    x = 1.0
    
    g = lambda x: x**2
    f = lambda u: np.exp(u)
    y = lambda x: f(g(x))
    
    dy_dx = numerical_derivative(y, x)
    theoretical = 2*x*np.exp(x**2)
    
    print(f"y = exp(x²) 在x={x}")
    print(f"数值导数: {dy_dx:.6f}")
    print(f"理论值: {theoretical:.6f}")

def test_partial_derivatives():
    print("\n" + "=" * 50)
    print("偏导数练习")
    print("=" * 50)
    
    def partial_derivative(f, x, i, h=1e-5):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        return (f(x_plus) - f(x_minus)) / (2 * h)
    
    def gradient(f, x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            grad[i] = partial_derivative(f, x, i, h)
        return grad
    
    print("\n练习1: 计算偏导数")
    f = lambda x: x[0]**2 + x[1]**2 + x[0]*x[1]
    x = np.array([1.0, 2.0])
    
    df_dx0 = partial_derivative(f, x, 0)
    df_dx1 = partial_derivative(f, x, 1)
    
    print(f"f(x0, x1) = x0² + x1² + x0*x1")
    print(f"在点 (1, 2):")
    print(f"∂f/∂x0: 数值={df_dx0:.6f}, 理论={2*1 + 2:.6f}")
    print(f"∂f/∂x1: 数值={df_dx1:.6f}, 理论={2*2 + 1:.6f}")
    
    print("\n练习2: 计算梯度")
    grad = gradient(f, x)
    print(f"梯度: {grad}")

def test_gradient_descent():
    print("\n" + "=" * 50)
    print("梯度下降练习")
    print("=" * 50)
    
    print("\n练习1: 一元函数最小化")
    def f(x):
        return x**2
    
    def df(x):
        return 2*x
    
    x = 10.0
    learning_rate = 0.1
    history = [x]
    
    for i in range(50):
        x = x - learning_rate * df(x)
        history.append(x)
        if i % 10 == 0:
            print(f"迭代 {i}: x={x:.6f}, f(x)={f(x):.6f}")
    
    print(f"\n最终结果: x={x:.6f}, f(x)={f(x):.6f}")
    
    print("\n练习2: 多元函数最小化")
    def f_multi(x):
        return x[0]**2 + x[1]**2
    
    def grad_f_multi(x):
        return np.array([2*x[0], 2*x[1]])
    
    x = np.array([5.0, 3.0])
    learning_rate = 0.1
    history = [x.copy()]
    
    for i in range(50):
        grad = grad_f_multi(x)
        x = x - learning_rate * grad
        history.append(x.copy())
        if i % 10 == 0:
            print(f"迭代 {i}: x={x}, f(x)={f_multi(x):.6f}")
    
    print(f"\n最终结果: x={x}, f(x)={f_multi(x):.6f}")

def test_integration():
    print("\n" + "=" * 50)
    print("积分练习")
    print("=" * 50)
    
    print("\n练习1: 数值积分")
    def numerical_integral(f, a, b, n=1000):
        x = np.linspace(a, b, n)
        dx = (b - a) / (n - 1)
        return np.sum(f(x)) * dx
    
    f = lambda x: x**2
    a, b = 0, 1
    
    integral = numerical_integral(f, a, b)
    theoretical = (b**3 - a**3) / 3
    
    print(f"∫₀¹ x² dx")
    print(f"数值积分: {integral:.6f}")
    print(f"理论值: {theoretical:.6f}")
    
    print("\n练习2: 高斯积分")
    from scipy import integrate
    
    f = lambda x: np.exp(-x**2)
    result, error = integrate.quad(f, -np.inf, np.inf)
    
    print(f"∫₋∞^∞ e^(-x²) dx")
    print(f"数值结果: {result:.6f}")
    print(f"理论值: {np.sqrt(np.pi):.6f}")

def test_backpropagation():
    print("\n" + "=" * 50)
    print("反向传播练习")
    print("=" * 50)
    
    print("\n练习: 简单神经网络的反向传播")
    np.random.seed(42)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    learning_rate = 1.0
    
    for epoch in range(2000):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        
        loss = np.mean((a2 - y)**2)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        dz2 = (a2 - y) * sigmoid_derivative(a2)
        dW2 = a1.T @ dz2 / len(X)
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * sigmoid_derivative(a1)
        dW1 = X.T @ dz1 / len(X)
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
    
    print(f"\n最终预测:")
    print(a2.round(3))

def calculus_challenges():
    print("\n" + "=" * 50)
    print("微积分挑战题")
    print("=" * 50)
    
    print("\n挑战1: 实现Adam优化器")
    def adam_optimizer(grad_f, x_init, learning_rate=0.001, 
                       beta1=0.9, beta2=0.999, epsilon=1e-8, 
                       max_iter=1000):
        x = x_init.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        for t in range(1, max_iter + 1):
            grad = grad_f(x)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return x
    
    f = lambda x: x[0]**2 + x[1]**2
    grad_f = lambda x: np.array([2*x[0], 2*x[1]])
    
    x_init = np.array([10.0, 10.0])
    x_min = adam_optimizer(grad_f, x_init)
    
    print(f"初始点: {x_init}")
    print(f"最小值点: {x_min}")
    print(f"最小值: {f(x_min):.10f}")
    
    print("\n挑战2: 数值Hessian矩阵")
    def hessian(f, x, h=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
        
        return H
    
    f = lambda x: x[0]**2 + x[1]**2 + x[0]*x[1]
    x = np.array([1.0, 2.0])
    
    H = hessian(f, x)
    print(f"函数: f(x) = x0² + x1² + x0*x1")
    print(f"在点 {x} 的Hessian矩阵:")
    print(H)
    print(f"理论值: [[2, 1], [1, 2]]")

if __name__ == "__main__":
    test_derivatives()
    test_partial_derivatives()
    test_gradient_descent()
    test_integration()
    test_backpropagation()
    calculus_challenges()
    print("\n" + "=" * 50)
    print("微积分练习完成!")
    print("=" * 50)
