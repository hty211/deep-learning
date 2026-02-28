import torch
import torch.nn as nn
import math

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
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

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
            delta = (self.lora.lora_B @ self.lora.lora_A.T) * self.lora.scaling
            self.original_layer.weight.add_(delta.T)
        return self

class SimpleTransformerWithLoRA(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, rank=8, alpha=16):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.ModuleDict({
                    'q_proj': nn.Linear(hidden_size, hidden_size),
                    'k_proj': nn.Linear(hidden_size, hidden_size),
                    'v_proj': nn.Linear(hidden_size, hidden_size),
                    'o_proj': nn.Linear(hidden_size, hidden_size)
                }),
                'ffn': nn.ModuleDict({
                    'gate_proj': nn.Linear(hidden_size, hidden_size * 4),
                    'up_proj': nn.Linear(hidden_size, hidden_size * 4),
                    'down_proj': nn.Linear(hidden_size * 4, hidden_size)
                }),
                'norm1': nn.LayerNorm(hidden_size),
                'norm2': nn.LayerNorm(hidden_size)
            })
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        self._apply_lora(rank, alpha)
    
    def _apply_lora(self, rank, alpha):
        for layer in self.layers:
            layer['attention']['q_proj'] = LoRALinear(
                layer['attention']['q_proj'], rank, alpha
            )
            layer['attention']['v_proj'] = LoRALinear(
                layer['attention']['v_proj'], rank, alpha
            )
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            attn_out = self._attention(layer['attention'], x)
            x = layer['norm1'](x + attn_out)
            
            ffn_out = self._ffn(layer['ffn'], x)
            x = layer['norm2'](x + ffn_out)
        
        x = self.norm(x)
        return self.lm_head(x)
    
    def _attention(self, attn, x):
        q = attn['q_proj'](x)
        k = attn['k_proj'](x)
        v = attn['v_proj'](x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        return attn['o_proj'](out)
    
    def _ffn(self, ffn, x):
        gate = torch.sigmoid(ffn['gate_proj'](x))
        up = ffn['up_proj'](x)
        return ffn['down_proj'](gate * up)
    
    def get_trainable_params(self):
        trainable = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                trainable.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        return trainable
    
    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数: {total:,}")
        print(f"可训练参数: {trainable:,}")
        print(f"可训练比例: {100 * trainable / total:.4f}%")

def create_sample_data(vocab_size, num_samples=100, seq_len=32):
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    return inputs, targets

def main():
    print("=" * 60)
    print("LoRA微调示例")
    print("=" * 60)
    
    vocab_size = 1000
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    rank = 8
    alpha = 16
    
    model = SimpleTransformerWithLoRA(
        vocab_size, hidden_size, num_layers, num_heads, rank, alpha
    )
    
    print("\n模型配置:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  隐藏维度: {hidden_size}")
    print(f"  层数: {num_layers}")
    print(f"  LoRA Rank: {rank}")
    print(f"  LoRA Alpha: {alpha}")
    
    print("\n参数统计:")
    model.print_trainable_params()
    
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    inputs, targets = create_sample_data(vocab_size, num_samples=100)
    
    print("\n开始训练...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        batch_size = 10
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(batch_inputs)
            loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(inputs) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    print("\n测试推理...")
    model.eval()
    test_input = torch.randint(0, vocab_size, (1, 16))
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n合并LoRA权重...")
    for layer in model.layers:
        if isinstance(layer['attention']['q_proj'], LoRALinear):
            layer['attention']['q_proj'].merge_weights()
        if isinstance(layer['attention']['v_proj'], LoRALinear):
            layer['attention']['v_proj'].merge_weights()
    
    print("LoRA权重已合并到原始权重")
    
    print("\n" + "=" * 60)
    print("LoRA微调示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
