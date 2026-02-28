import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        
        assert embed_size % num_heads == 0
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        
        return self.out_proj(output), attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(nn.functional.gelu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.ffn = FeedForward(embed_size, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.ffn = FeedForward(embed_size, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, memory_mask=None):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        attn_out, _ = self.cross_attn(x, encoder_out, encoder_out, memory_mask)
        x = self.norm2(x + self.dropout2(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask
    
    def forward(self, x, encoder_out, tgt_mask=None, memory_mask=None):
        seq_len = x.size(1)
        
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len, x.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask, memory_mask)
        
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=512, 
                 num_layers=6, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, embed_size, num_layers, 
                                          num_heads, ff_dim, dropout=dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_size, num_layers,
                                          num_heads, ff_dim, dropout=dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, memory_mask)
        return decoder_out

def main():
    print("=" * 60)
    print("Transformer实现示例")
    print("=" * 60)
    
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embed_size = 256
    num_layers = 4
    num_heads = 8
    ff_dim = 512
    
    model = Transformer(src_vocab_size, tgt_vocab_size, embed_size, 
                       num_layers, num_heads, ff_dim)
    
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n模型配置:")
    print(f"  源词汇表大小: {src_vocab_size}")
    print(f"  目标词汇表大小: {tgt_vocab_size}")
    print(f"  嵌入维度: {embed_size}")
    print(f"  层数: {num_layers}")
    print(f"  注意力头数: {num_heads}")
    print(f"  前馈网络维度: {ff_dim}")
    
    output = model(src, tgt)
    
    print(f"\n输入形状:")
    print(f"  源序列: {src.shape}")
    print(f"  目标序列: {tgt.shape}")
    print(f"\n输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("Transformer实现完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
