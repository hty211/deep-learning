import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        out, hn = self.rnn(x, h0)
        
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, hn = self.gru(x, h0)
        
        out = self.fc(out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        return self.fc(context)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_models():
    print("=" * 50)
    print("RNN模型测试")
    print("=" * 50)
    
    batch_size = 16
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 10
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("\n1. SimpleRNN")
    model = SimpleRNN(input_size, hidden_size, output_size)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n2. SimpleLSTM")
    model = SimpleLSTM(input_size, hidden_size, output_size)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n3. SimpleGRU")
    model = SimpleGRU(input_size, hidden_size, output_size)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n4. BiLSTM")
    model = BiLSTM(input_size, hidden_size, output_size)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n5. StackedLSTM (3层)")
    model = StackedLSTM(input_size, hidden_size, output_size, num_layers=3)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n6. LSTMWithAttention")
    model = LSTMWithAttention(input_size, hidden_size, output_size)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")

def compare_rnn_types():
    print("\n" + "=" * 50)
    print("RNN类型对比")
    print("=" * 50)
    
    input_size = 32
    hidden_size = 64
    output_size = 10
    
    models = {
        'RNN': SimpleRNN(input_size, hidden_size, output_size),
        'LSTM': SimpleLSTM(input_size, hidden_size, output_size),
        'GRU': SimpleGRU(input_size, hidden_size, output_size),
        'BiLSTM': BiLSTM(input_size, hidden_size, output_size),
    }
    
    print("\n参数量对比:")
    for name, model in models.items():
        params = count_parameters(model)
        print(f"  {name}: {params:,}")

def demonstrate_hidden_states():
    print("\n" + "=" * 50)
    print("隐藏状态演示")
    print("=" * 50)
    
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 16
    
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    h0 = torch.zeros(1, batch_size, hidden_size)
    c0 = torch.zeros(1, batch_size, hidden_size)
    
    output, (hn, cn) = lstm(x, (h0, c0))
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"最终隐藏状态形状: {hn.shape}")
    print(f"最终细胞状态形状: {cn.shape}")
    
    print("\n每个时间步的输出:")
    for t in range(seq_len):
        print(f"  时间步 {t}: {output[0, t, :4].tolist()}...")

def demonstrate_sequence_processing():
    print("\n" + "=" * 50)
    print("序列处理演示")
    print("=" * 50)
    
    vocab_size = 100
    embed_size = 32
    hidden_size = 64
    output_size = 10
    
    embedding = nn.Embedding(vocab_size, embed_size)
    lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    fc = nn.Linear(hidden_size, output_size)
    
    text = torch.randint(0, vocab_size, (2, 8))
    
    print(f"输入文本序列: {text.shape}")
    
    embedded = embedding(text)
    print(f"嵌入后: {embedded.shape}")
    
    lstm_out, _ = lstm(embedded)
    print(f"LSTM输出: {lstm_out.shape}")
    
    final_out = fc(lstm_out[:, -1, :])
    print(f"最终输出: {final_out.shape}")

if __name__ == "__main__":
    test_models()
    compare_rnn_types()
    demonstrate_hidden_states()
    demonstrate_sequence_processing()
    print("\n" + "=" * 50)
    print("RNN基础示例完成!")
    print("=" * 50)
