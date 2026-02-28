# BERT原理

## 1. BERT概述

```python
import torch
import torch.nn as nn

def explain_bert():
    print("BERT: Bidirectional Encoder Representations from Transformers")
    print("\n核心特点:")
    print("  1. 双向编码: 同时看到左右上下文")
    print("  2. 预训练+微调: 大规模无监督预训练")
    print("  3. 统一架构: 一个模型处理多种任务")

explain_bert()
```

## 2. BERT架构

### 2.1 模型结构

```python
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size=2, dropout=0.1):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

print("BERT输入嵌入组成:")
print("  1. Token Embeddings: 词嵌入")
print("  2. Segment Embeddings: 句子嵌入")
print("  3. Position Embeddings: 位置嵌入")
```

### 2.2 BERT编码器

```python
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(self.attention_output(attention_output))
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        intermediate_output = nn.functional.gelu(self.intermediate(hidden_states))
        layer_output = self.dropout(self.output(intermediate_output))
        hidden_states = self.output_norm(hidden_states + layer_output)
        
        return hidden_states
```

### 2.3 完整BERT模型

```python
class BertModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, 
                 num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))
        
        return hidden_states, pooled_output

print("\nBERT模型变体:")
print("  BERT-Base: 12层, 768维, 12头, 110M参数")
print("  BERT-Large: 24层, 1024维, 16头, 340M参数")
```

## 3. 预训练任务

### 3.1 掩码语言模型 (MLM)

```python
class BertMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        return logits

def create_mlm_mask(input_ids, vocab_size, mask_prob=0.15):
    labels = input_ids.clone()
    
    probability_matrix = torch.full(labels.shape, mask_prob)
    
    mask = torch.bernoulli(probability_matrix).bool()
    
    labels[~mask] = -100
    
    masked_input = input_ids.clone()
    
    mask_indices = mask
    masked_input[mask_indices] = vocab_size - 1
    
    replace_10_percent = mask_indices & (torch.rand(input_ids.shape) < 0.1)
    masked_input[replace_10_percent] = torch.randint(0, vocab_size - 1, replace_10_percent.sum().item())
    
    return masked_input, labels

print("\nMLM任务:")
print("  1. 随机遮蔽15%的token")
print("  2. 80%替换为[MASK]")
print("  3. 10%替换为随机词")
print("  4. 10%保持不变")
print("  5. 预测被遮蔽的词")
```

### 3.2 下一句预测 (NSP)

```python
class BertNSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, pooled_output):
        return self.classifier(pooled_output)

def create_nsp_data(sentences_a, sentences_b, is_next=True):
    """
    创建NSP训练数据
    """
    if is_next:
        label = 1
    else:
        label = 0
    
    input_ids = ['[CLS]'] + sentences_a + ['[SEP]'] + sentences_b + ['[SEP]']
    token_type_ids = [0] * (len(sentences_a) + 2) + [1] * (len(sentences_b) + 1)
    
    return input_ids, token_type_ids, label

print("\nNSP任务:")
print("  输入: [CLS] 句子A [SEP] 句子B [SEP]")
print("  输出: IsNext(1) 或 NotNext(0)")
print("  训练: 50%正样本, 50%负样本")
```

## 4. 输入格式

```python
def explain_bert_input():
    print("\nBERT输入格式:")
    print("  [CLS] Token A Token B [SEP] Token C Token D [SEP]")
    print("    |      |              |      |              |")
    print("  Segment: 0  0  0  0  0  1  1  1  1  1")
    print("\n特殊标记:")
    print("  [CLS]: 句子开始，用于分类任务")
    print("  [SEP]: 句子分隔")
    print("  [MASK]: 遮蔽标记，用于MLM")
    print("  [PAD]: 填充标记")
    print("  [UNK]: 未知词标记")

explain_bert_input()
```

## 5. 微调策略

### 5.1 分类任务

```python
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.pooler.out_features, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

print("\n分类任务微调:")
print("  输入: [CLS] 文本 [SEP]")
print("  取[CLS]位置的输出")
print("  接分类层 -> logits")
```

### 5.2 序列标注任务

```python
class BertForTokenClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.layers[0].attention.all_head_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hidden_states, _ = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

print("\n序列标注任务微调:")
print("  输入: [CLS] Token1 Token2 ... [SEP]")
print("  取每个Token位置的输出")
print("  接分类层 -> 每个Token的标签")
```

### 5.3 问答任务

```python
class BertForQuestionAnswering(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        
        self.bert = bert_model
        self.qa_outputs = nn.Linear(bert_model.layers[0].attention.all_head_size, 2)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hidden_states, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(hidden_states)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

print("\n问答任务微调:")
print("  输入: [CLS] 问题 [SEP] 文章 [SEP]")
print("  预测答案起始和结束位置")
```

## 6. BERT变体

```python
def explain_bert_variants():
    print("\nBERT变体:")
    print("\n1. RoBERTa")
    print("   - 更大数据集、更长训练")
    print("   - 去掉NSP任务")
    print("   - 动态遮蔽")
    
    print("\n2. ALBERT")
    print("   - 参数共享减少参数量")
    print("   - 因式分解嵌入")
    
    print("\n3. DistilBERT")
    print("   - 知识蒸馏压缩模型")
    print("   - 减少40%参数，保持97%性能")
    
    print("\n4. ELECTRA")
    print("   - 替换词检测任务")
    print("   - 更高效的预训练")
    
    print("\n5. DeBERTa")
    print("   - 解耦注意力机制")
    print("   - 增强掩码解码")

explain_bert_variants()
```

## 7. 总结

| 组件 | 说明 |
|------|------|
| 输入嵌入 | Token + Segment + Position |
| 编码器 | 多层Transformer编码器 |
| MLM | 遮蔽语言模型预训练 |
| NSP | 下一句预测预训练 |
| 微调 | 下游任务适配 |
