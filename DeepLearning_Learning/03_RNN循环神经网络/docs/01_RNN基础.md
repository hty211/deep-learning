# RNN基础原理

## 1. 什么是RNN

循环神经网络(Recurrent Neural Network)是一种处理序列数据的神经网络。

### 1.1 序列数据

```python
import numpy as np

text_sequence = ["我", "爱", "学习", "深度", "学习"]
time_series = np.sin(np.linspace(0, 10, 100))

print(f"文本序列长度: {len(text_sequence)}")
print(f"时间序列长度: {len(time_series)}")
```

### 1.2 RNN的基本结构

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.hidden_states = [h]
        self.inputs = inputs
        
        outputs = []
        for x in inputs:
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            outputs.append(y)
            self.hidden_states.append(h)
        
        return outputs, h
    
    def backward(self, d_outputs, learning_rate=0.01):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(len(self.inputs))):
            dy = d_outputs[t]
            dWhy += dy @ self.hidden_states[t+1].T
            dby += dy
            
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - self.hidden_states[t+1] ** 2) * dh
            
            dWxh += dh_raw @ self.inputs[t].T
            dWhh += dh_raw @ self.hidden_states[t].T
            dbh += dh_raw
            
            dh_next = self.Whh.T @ dh_raw
        
        for param, dparam in [(self.Wxh, dWxh), (self.Whh, dWhh), 
                               (self.Why, dWhy), (self.bh, dbh), (self.by, dby)]:
            np.clip(dparam, -5, 5, out=dparam)
            param -= learning_rate * dparam

input_size = 3
hidden_size = 4
output_size = 2

rnn = SimpleRNN(input_size, hidden_size, output_size)

seq_len = 5
inputs = [np.random.randn(input_size, 1) for _ in range(seq_len)]

outputs, final_hidden = rnn.forward(inputs)
print(f"序列长度: {seq_len}")
print(f"每个时间步输出形状: {outputs[0].shape}")
print(f"最终隐藏状态形状: {final_hidden.shape}")
```

## 2. RNN的计算过程

### 2.1 前向传播

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y
```

```python
import numpy as np

def rnn_forward(X, Wxh, Whh, Why, bh, by, h0):
    """
    X: 输入序列 (seq_len, input_size)
    h0: 初始隐藏状态 (hidden_size,)
    """
    seq_len = X.shape[0]
    hidden_size = Whh.shape[0]
    
    H = np.zeros((seq_len + 1, hidden_size))
    H[0] = h0
    
    Y = []
    
    for t in range(seq_len):
        H[t+1] = np.tanh(X[t] @ Wxh.T + H[t] @ Whh.T + bh)
        y = H[t+1] @ Why.T + by
        Y.append(y)
    
    return np.array(Y), H[1:]

np.random.seed(42)
seq_len, input_size, hidden_size, output_size = 5, 3, 4, 2

X = np.random.randn(seq_len, input_size)
Wxh = np.random.randn(hidden_size, input_size) * 0.1
Whh = np.random.randn(hidden_size, hidden_size) * 0.1
Why = np.random.randn(output_size, hidden_size) * 0.1
bh = np.zeros(hidden_size)
by = np.zeros(output_size)
h0 = np.zeros(hidden_size)

Y, H = rnn_forward(X, Wxh, Whh, Why, bh, by, h0)

print(f"输入形状: {X.shape}")
print(f"输出形状: {Y.shape}")
print(f"隐藏状态形状: {H.shape}")
```

### 2.2 随时间反向传播(BPTT)

```python
def rnn_backward(X, Y_true, Y_pred, H, Wxh, Whh, Why, bh, by, h0, learning_rate=0.01):
    seq_len = X.shape[0]
    hidden_size = Whh.shape[0]
    
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    
    dH_next = np.zeros(hidden_size)
    
    for t in reversed(range(seq_len)):
        dY = 2 * (Y_pred[t] - Y_true[t])
        
        dWhy += np.outer(dY, H[t])
        dby += dY
        
        dH = Why.T @ dY + dH_next
        dH_raw = dH * (1 - H[t] ** 2)
        
        dWxh += np.outer(dH_raw, X[t])
        dWhh += np.outer(dH_raw, H[t-1] if t > 0 else h0)
        dbh += dH_raw
        
        dH_next = Whh.T @ dH_raw
    
    for param, dparam in [(Wxh, dWxh), (Whh, dWhh), (Why, dWhy), (bh, dbh), (by, dby)]:
        np.clip(dparam, -5, 5, out=dparam)
        param -= learning_rate * dparam
    
    return Wxh, Whh, Why, bh, by
```

## 3. 梯度消失与梯度爆炸

### 3.1 问题演示

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_gradient_problem():
    seq_len = 50
    hidden_size = 10
    
    np.random.seed(42)
    Whh = np.random.randn(hidden_size, hidden_size) * 0.5
    
    eigenvalues = np.linalg.eigvals(Whh)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"最大特征值: {max_eigenvalue:.4f}")
    
    gradients = []
    grad = np.ones(hidden_size)
    
    for t in range(seq_len):
        grad = Whh.T @ grad
        gradients.append(np.linalg.norm(grad))
    
    plt.figure(figsize=(10, 4))
    plt.plot(gradients)
    plt.xlabel('时间步')
    plt.ylabel('梯度范数')
    plt.title('RNN梯度变化演示')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return gradients

gradients = demonstrate_gradient_problem()
print(f"初始梯度范数: {gradients[0]:.6f}")
print(f"最终梯度范数: {gradients[-1]:.6e}")
```

### 3.2 梯度裁剪

```python
def clip_gradient(grad, max_norm=5.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad

grad = np.random.randn(100) * 10
print(f"裁剪前梯度范数: {np.linalg.norm(grad):.4f}")

grad_clipped = clip_gradient(grad, max_norm=5.0)
print(f"裁剪后梯度范数: {np.linalg.norm(grad_clipped):.4f}")
```

## 4. PyTorch中的RNN

### 4.1 基本RNN

```python
import torch
import torch.nn as nn

rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=False
)

batch_size = 3
seq_len = 5
x = torch.randn(batch_size, seq_len, 10)

h0 = torch.zeros(2, batch_size, 20)

output, hn = rnn(x, h0)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"最终隐藏状态形状: {hn.shape}")
```

### 4.2 自定义RNN模型

```python
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, hn = self.rnn(x, h0)
        
        out = self.fc(out[:, -1, :])
        
        return out, hn

model = RNNModel(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(3, 5, 10)
output, hn = model(x)

print(f"模型输出形状: {output.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 5. RNN的应用场景

| 应用 | 输入 | 输出 | 类型 |
|------|------|------|------|
| 文本分类 | 文本序列 | 类别标签 | 多对一 |
| 情感分析 | 评论序列 | 情感标签 | 多对一 |
| 机器翻译 | 源语言序列 | 目标语言序列 | 多对多 |
| 语音识别 | 音频序列 | 文本序列 | 多对多 |
| 时间序列预测 | 历史序列 | 未来值 | 多对多 |

## 6. RNN的变体

```python
class ManyToOneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hn = self.rnn(x)
        return self.fc(hn[-1])

class OneToManyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(h)
        return self.fc(out)

class ManyToManyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

print("RNN类型:")
print("1. Many-to-One: 序列输入 -> 单个输出 (文本分类)")
print("2. One-to-Many: 单个输入 -> 序列输出 (图像描述)")
print("3. Many-to-Many: 序列输入 -> 序列输出 (机器翻译)")
```

## 7. 总结

- RNN通过隐藏状态传递历史信息
- BPTT用于训练RNN
- 梯度消失/爆炸是RNN的主要问题
- 梯度裁剪可以缓解梯度爆炸
- 根据任务选择合适的RNN类型
