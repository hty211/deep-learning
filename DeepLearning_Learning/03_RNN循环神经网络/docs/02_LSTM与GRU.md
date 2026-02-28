# LSTM与GRU

## 1. LSTM (Long Short-Term Memory)

LSTM通过门控机制解决RNN的梯度消失问题。

### 1.1 LSTM结构

```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, prev_h, prev_c):
        concat = np.vstack((prev_h, x))
        
        ft = self.sigmoid(self.Wf @ concat + self.bf)
        it = self.sigmoid(self.Wi @ concat + self.bi)
        cct = np.tanh(self.Wc @ concat + self.bc)
        ot = self.sigmoid(self.Wo @ concat + self.bo)
        
        c = ft * prev_c + it * cct
        h = ot * np.tanh(c)
        
        return h, c, (ft, it, cct, ot)
    
    def forward_sequence(self, X):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        outputs = []
        for x in X:
            x = x.reshape(-1, 1)
            h, c, _ = self.forward(x, h, c)
            outputs.append(h.flatten())
        
        return np.array(outputs)

input_size = 3
hidden_size = 4
seq_len = 5

lstm = LSTM(input_size, hidden_size)
X = [np.random.randn(input_size) for _ in range(seq_len)]

outputs = lstm.forward_sequence(X)
print(f"输入序列长度: {seq_len}")
print(f"输出形状: {outputs.shape}")
```

### 1.2 LSTM门控机制

```python
def explain_lstm_gates():
    print("LSTM三个门的作用:")
    print("\n1. 遗忘门 (Forget Gate):")
    print("   f_t = σ(W_f * [h_{t-1}, x_t] + b_f)")
    print("   决定丢弃哪些信息")
    
    print("\n2. 输入门 (Input Gate):")
    print("   i_t = σ(W_i * [h_{t-1}, x_t] + b_i)")
    print("   Ĉ_t = tanh(W_C * [h_{t-1}, x_t] + b_C)")
    print("   决定存储哪些新信息")
    
    print("\n3. 输出门 (Output Gate):")
    print("   o_t = σ(W_o * [h_{t-1}, x_t] + b_o)")
    print("   h_t = o_t * tanh(C_t)")
    print("   决定输出哪些信息")
    
    print("\n细胞状态更新:")
    print("   C_t = f_t * C_{t-1} + i_t * Ĉ_t")

explain_lstm_gates()
```

### 1.3 PyTorch LSTM

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=False,
    dropout=0.1
)

batch_size = 3
seq_len = 5
x = torch.randn(batch_size, seq_len, 10)

h0 = torch.zeros(2, batch_size, 20)
c0 = torch.zeros(2, batch_size, 20)

output, (hn, cn) = lstm(x, (h0, c0))

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"隐藏状态形状: {hn.shape}")
print(f"细胞状态形状: {cn.shape}")
```

## 2. GRU (Gated Recurrent Unit)

GRU是LSTM的简化版本，参数更少。

### 2.1 GRU结构

```python
class GRU:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, prev_h):
        concat = np.vstack((prev_h, x))
        
        zt = self.sigmoid(self.Wz @ concat + self.bz)
        rt = self.sigmoid(self.Wr @ concat + self.br)
        
        concat_r = np.vstack((rt * prev_h, x))
        ht = np.tanh(self.Wh @ concat_r + self.bh)
        
        h = (1 - zt) * prev_h + zt * ht
        
        return h
    
    def forward_sequence(self, X):
        h = np.zeros((self.hidden_size, 1))
        
        outputs = []
        for x in X:
            x = x.reshape(-1, 1)
            h = self.forward(x, h)
            outputs.append(h.flatten())
        
        return np.array(outputs)

gru = GRU(input_size=3, hidden_size=4)
X = [np.random.randn(3) for _ in range(5)]
outputs = gru.forward_sequence(X)
print(f"GRU输出形状: {outputs.shape}")
```

### 2.2 GRU门控机制

```python
def explain_gru_gates():
    print("GRU两个门的作用:")
    
    print("\n1. 更新门 (Update Gate):")
    print("   z_t = σ(W_z * [h_{t-1}, x_t])")
    print("   决定保留多少旧信息和添加多少新信息")
    
    print("\n2. 重置门 (Reset Gate):")
    print("   r_t = σ(W_r * [h_{t-1}, x_t])")
    print("   决定忽略多少过去信息")
    
    print("\n候选隐藏状态:")
    print("   h̃_t = tanh(W * [r_t * h_{t-1}, x_t])")
    
    print("\n最终隐藏状态:")
    print("   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t")

explain_gru_gates()
```

### 2.3 PyTorch GRU

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=False,
    dropout=0.1
)

x = torch.randn(3, 5, 10)
h0 = torch.zeros(2, 3, 20)

output, hn = gru(x, h0)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"隐藏状态形状: {hn.shape}")
```

## 3. LSTM vs GRU vs RNN

```python
import torch
import torch.nn as nn
import time

def compare_models():
    input_size = 128
    hidden_size = 256
    num_layers = 2
    seq_len = 50
    batch_size = 32
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    
    models = {'RNN': rnn, 'LSTM': lstm, 'GRU': gru}
    
    print("模型对比:")
    print("-" * 50)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        
        start = time.time()
        for _ in range(10):
            _ = model(x)
        elapsed = time.time() - start
        
        print(f"{name}:")
        print(f"  参数量: {params:,}")
        print(f"  前向传播时间 (10次): {elapsed:.4f}s")
        print()

compare_models()
```

## 4. 双向RNN

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        
        out = self.fc(output[:, -1, :])
        return out

model = BiLSTM(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(3, 5, 10)
output = model(x)

print(f"双向LSTM输入: {x.shape}")
print(f"双向LSTM输出: {output.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 5. 深层RNN

```python
class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

for num_layers in [1, 2, 3, 4]:
    model = DeepLSTM(10, 20, 5, num_layers)
    params = sum(p.numel() for p in model.parameters())
    print(f"层数={num_layers}, 参数量={params:,}")
```

## 6. 实际应用示例

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, 
                 num_layers=2, rnn_type='lstm'):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True, dropout=0.3)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True, dropout=0.3)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])

vocab_size = 10000
embed_size = 128
hidden_size = 256
output_size = 2

for rnn_type in ['rnn', 'gru', 'lstm']:
    model = TextClassifier(vocab_size, embed_size, hidden_size, output_size, 
                           rnn_type=rnn_type)
    params = sum(p.numel() for p in model.parameters())
    print(f"{rnn_type.upper()}模型参数量: {params:,}")
```

## 7. 总结

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 门数量 | 0 | 3 | 2 |
| 参数量 | 最少 | 最多 | 中等 |
| 训练速度 | 最快 | 最慢 | 中等 |
| 长序列能力 | 差 | 好 | 好 |
| 适用场景 | 短序列 | 长序列 | 长序列(效率优先) |

选择建议：
- 短序列：简单RNN
- 长序列：LSTM或GRU
- 计算资源有限：GRU
- 需要最佳性能：LSTM
