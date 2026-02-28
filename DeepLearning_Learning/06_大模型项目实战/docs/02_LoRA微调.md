# LoRA微调

## 1. LoRA原理

```python
import torch
import torch.nn as nn
import math

def explain_lora():
    print("LoRA (Low-Rank Adaptation) 核心思想:")
    print("  模型适应过程中的权重变化具有低秩特性")
    print("  W' = W + ΔW = W + BA")
    print("  其中 B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)")

explain_lora()
```

## 2. LoRA实现

### 2.1 基础LoRA层

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        result = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank, alpha, dropout
        )
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)
    
    def merge_weights(self):
        with torch.no_grad():
            self.original_layer.weight.add_(
                self.lora.lora_B.T @ self.lora.lora_A.T * self.lora.scaling
            )

print("LoRA Linear层:")
print("  - 保留原始权重冻结")
print("  - 添加低秩适配器")
print("  - 前向传播: output = Wx + BAx * scaling")
print("  - 可合并: W' = W + BA * scaling")
```

### 2.2 LoRA注意力模块

```python
class LoRAAttention(nn.Module):
    def __init__(self, original_attn, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        
        self.original_attn = original_attn
        
        hidden_size = original_attn.hidden_size if hasattr(original_attn, 'hidden_size') else 768
        
        self.q_lora = LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
        self.v_lora = LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        query = self.original_attn.q_proj(hidden_states) + self.q_lora(hidden_states)
        key = self.original_attn.k_proj(hidden_states)
        value = self.original_attn.v_proj(hidden_states) + self.v_lora(hidden_states)
        
        return query, key, value

print("\nLoRA注意力:")
print("  - 通常只对Q和V应用LoRA")
print("  - 减少参数量")
print("  - 实验证明效果良好")
```

## 3. LoRA配置

### 3.1 参数选择

```python
def lora_config_guide():
    print("LoRA参数配置指南:")
    print("\n1. Rank (秩):")
    print("   - 常用值: 4, 8, 16, 32, 64")
    print("   - 较小任务: rank=4~8")
    print("   - 复杂任务: rank=16~64")
    print("   - 更大的rank不一定更好")
    
    print("\n2. Alpha (缩放系数):")
    print("   - 常用值: 16, 32")
    print("   - 通常 alpha = 2 * rank")
    print("   - 控制LoRA的影响强度")
    
    print("\n3. Target Modules:")
    print("   - 常用: q_proj, v_proj")
    print("   - 更强效果: q_proj, k_proj, v_proj, o_proj")
    print("   - 全量: 所有线性层")
    
    print("\n4. Dropout:")
    print("   - 常用值: 0.05, 0.1")
    print("   - 防止过拟合")

lora_config_guide()
```

### 3.2 配置示例

```python
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

def print_config(config):
    print("\nLoRA配置:")
    print(f"  Rank: {config.r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target Modules: {config.target_modules}")
    print(f"  Bias: {config.bias}")
    print(f"  Task Type: {config.task_type}")

config = LoRAConfig(r=16, lora_alpha=32)
print_config(config)
```

## 4. 训练流程

### 4.1 数据准备

```python
def prepare_data():
    print("微调数据准备:")
    print("\n1. 数据格式:")
    print("   - 指令微调: {'instruction': '', 'input': '', 'output': ''}")
    print("   - 对话格式: {'messages': [{'role': 'user', 'content': ''}, ...]}")
    
    print("\n2. 数据质量:")
    print("   - 清洗噪声数据")
    print("   - 确保标注准确")
    print("   - 保持多样性")
    
    print("\n3. 数据量:")
    print("   - 简单任务: 1000-5000条")
    print("   - 复杂任务: 10000-50000条")
    print("   - 质量 > 数量")

prepare_data()
```

### 4.2 训练代码

```python
class SimpleLoRATrainer:
    def __init__(self, model, train_data, config):
        self.model = model
        self.train_data = train_data
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            self.get_trainable_params(),
            lr=config.learning_rate
        )
    
    def get_trainable_params(self):
        trainable = []
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                trainable.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        return trainable
    
    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {100 * trainable_params / total_params:.4f}%")
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        outputs = self.model(**batch)
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

print("\n训练要点:")
print("  1. 只更新LoRA参数")
print("  2. 使用较小的学习率")
print("  3. 监控验证集性能")
print("  4. 保存LoRA权重而非完整模型")
```

## 5. 权重合并与导出

```python
def merge_lora_weights(model):
    print("LoRA权重合并:")
    print("  1. 遍历所有LoRA层")
    print("  2. 计算: W_new = W + (B @ A) * scaling")
    print("  3. 替换原始权重")
    print("  4. 移除LoRA模块")
    
    for name, module in model.named_modules():
        if hasattr(module, 'merge_weights'):
            module.merge_weights()
    
    return model

def export_lora(model, save_path):
    print(f"\n导出LoRA权重到: {save_path}")
    
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_state_dict[name] = param.data
    
    torch.save(lora_state_dict, save_path)
    print(f"  保存了 {len(lora_state_dict)} 个参数")

print("\n部署选项:")
print("  1. 合并后部署: 单一模型文件")
print("  2. 分开部署: 基础模型 + LoRA适配器")
print("  3. 多LoRA切换: 一个基础模型 + 多个LoRA")
```

## 6. 实践建议

```python
def best_practices():
    print("\nLoRA最佳实践:")
    print("\n1. 初始化:")
    print("   - A使用随机初始化")
    print("   - B初始化为零")
    print("   - 确保训练开始时LoRA输出为零")
    
    print("\n2. 超参数:")
    print("   - 学习率: 1e-4 ~ 5e-4")
    print("   - Batch Size: 尽可能大")
    print("   - Epochs: 3-5 (避免过拟合)")
    
    print("\n3. 监控:")
    print("   - 训练损失下降")
    print("   - 验证集性能")
    print("   - 生成样本质量")
    
    print("\n4. 调试:")
    print("   - 检查梯度流")
    print("   - 验证LoRA参数更新")
    print("   - 对比不同rank效果")

best_practices()
```

## 7. 总结

| 方面 | 建议 |
|------|------|
| Rank | 8-32，根据任务复杂度调整 |
| Alpha | 通常为2*rank |
| Target | q_proj, v_proj 开始 |
| 学习率 | 1e-4 ~ 5e-4 |
| 数据 | 质量 > 数量 |
| 部署 | 合并权重或分开加载 |
