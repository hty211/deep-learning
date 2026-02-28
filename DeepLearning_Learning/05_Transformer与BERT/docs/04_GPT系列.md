# GPT系列

## 1. GPT概述

```python
import torch
import torch.nn as nn
import math

def explain_gpt():
    print("GPT: Generative Pre-trained Transformer")
    print("\n核心特点:")
    print("  1. 自回归生成: 从左到右预测下一个词")
    print("  2. 单向注意力: 只能看到之前的词")
    print("  3. 大规模预训练: 大数据+大模型")

explain_gpt()
```

## 2. GPT架构

### 2.1 GPT模型结构

```python
class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        
        return embeddings

class GPTAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.c_attn = nn.Linear(embed_size, 3 * embed_size)
        self.c_proj = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.shape
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(embed_size, dim=2)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask[:, :, :seq_length, :seq_length] == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)
        
        output = self.c_proj(attn_output)
        return output

class GPTMLP(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.1):
        super().__init__()
        
        self.c_fc = nn.Linear(embed_size, hidden_size)
        self.c_proj = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = GPTAttention(embed_size, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = GPTMLP(embed_size, hidden_size, dropout)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x
```

### 2.2 完整GPT模型

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, 
                 hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        
        self.embeddings = GPTEmbeddings(vocab_size, embed_size, max_position_embeddings, dropout)
        
        self.layers = nn.ModuleList([
            GPTBlock(embed_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        
        self.lm_head.weight = self.embeddings.token_embedding.weight
    
    def forward(self, input_ids, mask=None):
        x = self.embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate_causal_mask(self, seq_length, device):
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

print("GPT模型变体:")
print("  GPT-1: 12层, 768维, 12头, 117M参数")
print("  GPT-2: 48层, 1600维, 25头, 1.5B参数")
print("  GPT-3: 96层, 12288维, 96头, 175B参数")
```

## 3. 文本生成

### 3.1 自回归生成

```python
class GPTGenerator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=None, top_p=None):
        self.model.eval()
        
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated)
                next_token_logits = outputs[0, -1, :]
                
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated[0])

print("\n生成策略:")
print("  1. Greedy: 每步选最大概率词")
print("  2. Beam Search: 保留多个候选序列")
print("  3. Top-k: 从概率最高的k个词中采样")
print("  4. Top-p (Nucleus): 从累积概率达到p的词中采样")
print("  5. Temperature: 控制随机性")
```

## 4. GPT系列演进

### 4.1 GPT-1

```python
def explain_gpt1():
    print("\nGPT-1 (2018):")
    print("  - 首次提出生成式预训练")
    print("  - 两阶段: 无监督预训练 + 有监督微调")
    print("  - 任务: 自然语言推理、问答、分类等")
    print("  - 参数: 117M")

explain_gpt1()
```

### 4.2 GPT-2

```python
def explain_gpt2():
    print("\nGPT-2 (2019):")
    print("  - 零样本学习能力")
    print("  - 更大模型和数据")
    print("  - WebText数据集(800万网页)")
    print("  - 参数: 1.5B")
    print("  - 特点: 任务描述作为Prompt")

explain_gpt2()
```

### 4.3 GPT-3

```python
def explain_gpt3():
    print("\nGPT-3 (2020):")
    print("  - 上下文学习(In-Context Learning)")
    print("  - 少样本学习(Few-Shot)")
    print("  - 参数: 175B")
    print("  - 训练数据: 45TB文本")
    print("  - 能力: 翻译、问答、代码生成等")

explain_gpt3()
```

### 4.4 GPT-4

```python
def explain_gpt4():
    print("\nGPT-4 (2023):")
    print("  - 多模态能力(文本+图像)")
    print("  - 更长的上下文窗口")
    print("  - 更强的推理能力")
    print("  - RLHF对齐")

explain_gpt4()
```

## 5. 提示工程

```python
def explain_prompt_engineering():
    print("\n提示工程技巧:")
    print("\n1. 零样本提示 (Zero-shot)")
    print("   示例: '将以下句子翻译成英文: 你好'")
    
    print("\n2. 少样本提示 (Few-shot)")
    print("   示例: ")
    print("   'Q: 1+1=? A: 2'")
    print("   'Q: 2+2=? A: 4'")
    print("   'Q: 3+3=? A:'")
    
    print("\n3. 思维链 (Chain-of-Thought)")
    print("   示例: '让我们一步步思考...'")
    
    print("\n4. 角色扮演")
    print("   示例: '你是一个专业的翻译专家...'")
    
    print("\n5. 结构化输出")
    print("   示例: '请以JSON格式输出...'")

explain_prompt_engineering()
```

## 6. GPT vs BERT

```python
def compare_gpt_bert():
    print("\nGPT vs BERT 对比:")
    print("\n| 特性 | GPT | BERT |")
    print("|------|-----|------|")
    print("| 架构 | 解码器 | 编码器 |")
    print("| 注意力 | 单向(因果) | 双向 |")
    print("| 预训练 | 自回归 | MLM+NSP |")
    print("| 应用 | 生成任务 | 理解任务 |")
    print("| 输出 | 续写文本 | 分类/标注 |")

compare_gpt_bert()
```

## 7. 总结

| 模型 | 参数 | 主要特点 |
|------|------|---------|
| GPT-1 | 117M | 预训练+微调范式 |
| GPT-2 | 1.5B | 零样本学习 |
| GPT-3 | 175B | 上下文学习 |
| GPT-4 | - | 多模态、强推理 |
