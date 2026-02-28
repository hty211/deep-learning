# Transformer架构

## 1. Transformer整体架构

```python
import torch
import torch.nn as nn
import math

def explain_transformer():
    print("Transformer架构组成:")
    print("  1. Encoder: 多层编码器堆叠")
    print("  2. Decoder: 多层解码器堆叠")
    print("  3. 输入嵌入 + 位置编码")
    print("  4. 输出层 + Softmax")

explain_transformer()
```

## 2. 编码器

### 2.1 编码器层

```python
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_size)
        )
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x

print("编码器层组成:")
print("  1. 多头自注意力")
print("  2. Add & Norm")
print("  3. 前馈网络")
print("  4. Add & Norm")
```

### 2.2 完整编码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, dropout=0.1, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = self._create_positional_encoding(max_len, embed_size)
        
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, embed_size):
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x, src_mask=None):
        seq_len = x.size(1)
        
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x
```

## 3. 解码器

### 3.1 解码器层

```python
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_size)
        )
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, memory_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        attn_out, _ = self.cross_attn(x, encoder_out, encoder_out, key_padding_mask=memory_mask)
        x = self.norm2(x + self.dropout2(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        
        return x

print("\n解码器层组成:")
print("  1. 掩码自注意力 (Masked Self-Attention)")
print("  2. Add & Norm")
print("  3. 交叉注意力 (Cross-Attention)")
print("  4. Add & Norm")
print("  5. 前馈网络")
print("  6. Add & Norm")
```

### 3.2 完整解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, dropout=0.1, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = self._create_positional_encoding(max_len, embed_size)
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, embed_size):
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x, encoder_out, tgt_mask=None, memory_mask=None):
        seq_len = x.size(1)
        
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len, x.device)
        
        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask, memory_mask)
        
        return self.fc_out(x)
```

## 4. 完整Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=512, 
                 num_layers=6, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, 
                              num_heads, ff_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_layers,
                              num_heads, ff_dim, dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, memory_mask)
        return decoder_out
    
    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_out, tgt_mask=None, memory_mask=None):
        return self.decoder(tgt, encoder_out, tgt_mask, memory_mask)

print("\nTransformer参数:")
print("  - embed_size: 512 (默认)")
print("  - num_layers: 6 (默认)")
print("  - num_heads: 8 (默认)")
print("  - ff_dim: 2048 (默认)")
print("  - dropout: 0.1 (默认)")
```

## 5. 前馈网络

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

print("\n前馈网络结构:")
print("  Linear(embed_size -> ff_dim)")
print("  ReLU")
print("  Dropout")
print("  Linear(ff_dim -> embed_size)")
```

## 6. 层归一化

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

print("\n层归一化 vs 批归一化:")
print("  层归一化: 在特征维度归一化")
print("  批归一化: 在批次维度归一化")
print("  Transformer使用层归一化的原因:")
print("    - 不依赖批次大小")
print("    - 适合变长序列")
print("    - 训练和推理行为一致")
```

## 7. 标签平滑

```python
class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, pred, target):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(pred, true_dist)

print("\n标签平滑作用:")
print("  - 防止模型过于自信")
print("  - 提高泛化能力")
print("  - 通常smoothing=0.1")
```

## 8. 训练技巧

```python
def transformer_training_tips():
    print("\nTransformer训练技巧:")
    print("  1. 学习率调度: Warmup + Decay")
    print("  2. 梯度裁剪: 防止梯度爆炸")
    print("  3. 标签平滑: 提高泛化")
    print("  4. Dropout: 防止过拟合")
    print("  5. 权重初始化: Xavier初始化")

transformer_training_tips()

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_count ** (-0.5),
            self.step_count * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

## 9. 推理与生成

```python
def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    model.eval()
    
    with torch.no_grad():
        encoder_out = model.encode(src)
    
    ys = torch.ones(1, 1).fill_(start_symbol).long()
    
    for _ in range(max_len - 1):
        with torch.no_grad():
            out = model.decode(ys, encoder_out)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).long()], dim=1)
        
        if next_word == end_symbol:
            break
    
    return ys

print("\n解码策略:")
print("  1. Greedy Search: 每步选最大概率词")
print("  2. Beam Search: 保留多个候选")
print("  3. Top-k Sampling: 从top-k中采样")
print("  4. Top-p Sampling: 从累积概率p中采样")
```

## 10. 总结

| 组件 | 功能 | 关键技术 |
|------|------|---------|
| 编码器 | 编码输入序列 | 自注意力 + FFN |
| 解码器 | 生成输出序列 | 掩码注意力 + 交叉注意力 |
| 位置编码 | 添加位置信息 | 正弦/余弦函数 |
| 层归一化 | 稳定训练 | 特征维度归一化 |
| 前馈网络 | 非线性变换 | 两层MLP |
