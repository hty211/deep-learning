import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def test_rnn_basics():
    print("=" * 50)
    print("RNN基础练习")
    print("=" * 50)
    
    print("\n练习1: RNN输出形状")
    rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    
    x = torch.randn(3, 5, 10)
    h0 = torch.zeros(2, 3, 20)
    
    output, hn = rnn(x, h0)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hn.shape}")
    
    print("\n练习2: LSTM输出形状")
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    
    h0 = torch.zeros(2, 3, 20)
    c0 = torch.zeros(2, 3, 20)
    
    output, (hn, cn) = lstm(x, (h0, c0))
    
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hn.shape}")
    print(f"细胞状态形状: {cn.shape}")
    
    print("\n练习3: 双向LSTM")
    bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, 
                     batch_first=True, bidirectional=True)
    
    output, _ = bilstm(x)
    print(f"双向LSTM输出形状: {output.shape}")

def test_sequence_handling():
    print("\n" + "=" * 50)
    print("序列处理练习")
    print("=" * 50)
    
    print("\n练习1: 变长序列处理")
    sequences = [
        torch.randn(3, 10),
        torch.randn(5, 10),
        torch.randn(4, 10),
    ]
    
    lengths = torch.tensor([3, 5, 4])
    
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    print(f"填充后形状: {padded.shape}")
    
    packed = nn.utils.rnn.pack_padded_sequence(
        padded, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    
    lstm = nn.LSTM(10, 20, batch_first=True)
    packed_output, _ = lstm(packed)
    
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    print(f"解包后输出形状: {output.shape}")
    
    print("\n练习2: 序列打包")
    print(f"原始序列长度: {[len(seq) for seq in sequences]}")
    print(f"填充后序列长度: {padded.shape[1]}")

def test_gradient_flow():
    print("\n" + "=" * 50)
    print("梯度流练习")
    print("=" * 50)
    
    print("\n练习: 梯度裁剪")
    model = nn.LSTM(10, 20, batch_first=True)
    
    x = torch.randn(2, 50, 10)
    y = torch.randint(0, 2, (2,))
    
    output, _ = model(x)
    pred = output[:, -1, :]
    
    loss = nn.CrossEntropyLoss()(pred, y)
    loss.backward()
    
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"裁剪前梯度范数: {total_norm:.4f}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"裁剪后梯度范数: {total_norm:.4f}")

def test_attention_mechanism():
    print("\n" + "=" * 50)
    print("注意力机制练习")
    print("=" * 50)
    
    print("\n练习: 实现简单注意力")
    
    class SimpleAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = nn.Linear(hidden_size, 1)
        
        def forward(self, lstm_output):
            attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
            context = torch.sum(lstm_output * attn_weights, dim=1)
            return context, attn_weights
    
    attention = SimpleAttention(20)
    
    lstm_output = torch.randn(2, 10, 20)
    context, attn_weights = attention(lstm_output)
    
    print(f"LSTM输出形状: {lstm_output.shape}")
    print(f"上下文向量形状: {context.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重和: {attn_weights.sum(dim=1)}")

def rnn_design_challenges():
    print("\n" + "=" * 50)
    print("RNN设计挑战")
    print("=" * 50)
    
    print("\n挑战1: 设计文本分类器")
    
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.fc(lstm_out[:, -1, :])
    
    model = TextClassifier(vocab_size=10000, embed_size=128, hidden_size=256, num_classes=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"文本分类器参数量: {params:,}")
    
    print("\n挑战2: 设计序列预测器")
    
    class SequencePredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out)
    
    model = SequencePredictor(input_size=10, hidden_size=20, output_size=10)
    x = torch.randn(2, 5, 10)
    output = model(x)
    print(f"序列预测器输入: {x.shape}")
    print(f"序列预测器输出: {output.shape}")
    
    print("\n挑战3: 设计多层双向LSTM")
    
    class DeepBiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True, dropout=0.2)
        
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    
    for num_layers in [1, 2, 3]:
        model = DeepBiLSTM(10, 20, num_layers)
        params = sum(p.numel() for p in model.parameters())
        print(f"层数={num_layers}, 参数量={params:,}")

def compare_rnn_variants():
    print("\n" + "=" * 50)
    print("RNN变体对比")
    print("=" * 50)
    
    input_size = 64
    hidden_size = 128
    seq_len = 20
    batch_size = 16
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    models = {
        'RNN': nn.RNN(input_size, hidden_size, batch_first=True),
        'LSTM': nn.LSTM(input_size, hidden_size, batch_first=True),
        'GRU': nn.GRU(input_size, hidden_size, batch_first=True),
    }
    
    print("\n参数量和输出形状对比:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        output, _ = model(x) if name != 'LSTM' else model(x)[0:1]
        if name == 'LSTM':
            output, _ = model(x)
        print(f"{name}: 参数量={params:,}, 输出形状={output.shape}")

if __name__ == "__main__":
    test_rnn_basics()
    test_sequence_handling()
    test_gradient_flow()
    test_attention_mechanism()
    rnn_design_challenges()
    compare_rnn_variants()
    print("\n" + "=" * 50)
    print("RNN练习完成!")
    print("=" * 50)
