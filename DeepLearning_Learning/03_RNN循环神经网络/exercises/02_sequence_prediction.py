import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_sine_wave(seq_length=100, num_samples=1000):
    t = np.linspace(0, 20 * np.pi, seq_length * num_samples)
    data = np.sin(t) + 0.1 * np.random.randn(len(t))
    data = data.reshape(num_samples, seq_length)
    return data

class SequencePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

class NextStepPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def test_sequence_prediction():
    print("=" * 50)
    print("序列预测练习")
    print("=" * 50)
    
    print("\n练习1: 正弦波预测")
    
    np.random.seed(42)
    data = generate_sine_wave(seq_length=50, num_samples=200)
    
    train_data = data[:150]
    test_data = data[150:]
    
    seq_len = 40
    X_train, y_train = [], []
    for sample in train_data:
        for i in range(len(sample) - seq_len):
            X_train.append(sample[i:i+seq_len])
            y_train.append(sample[i+1:i+seq_len+1])
    
    X_train = torch.FloatTensor(np.array(X_train)).unsqueeze(-1)
    y_train = torch.FloatTensor(np.array(y_train)).unsqueeze(-1)
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"目标数据形状: {y_train.shape}")
    
    model = SequencePredictor(input_size=1, hidden_size=32, num_layers=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练模型...")
    losses = []
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X_train[:100])
        loss = criterion(output, y_train[:100])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.savefig('sequence_loss.png', dpi=150)
    plt.show()

def test_next_step_prediction():
    print("\n" + "=" * 50)
    print("下一步预测练习")
    print("=" * 50)
    
    print("\n练习: 预测序列的下一个值")
    
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 500)
    data = np.sin(t)
    
    seq_len = 20
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    model = NextStepPredictor(input_size=1, hidden_size=32, num_layers=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练模型...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test)
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.numpy()[:100], label='真实值')
    plt.plot(predictions[:100], label='预测值')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.title('下一步预测结果')
    plt.legend()
    plt.grid(True)
    plt.savefig('next_step_prediction.png', dpi=150)
    plt.show()

def test_multi_step_prediction():
    print("\n" + "=" * 50)
    print("多步预测练习")
    print("=" * 50)
    
    print("\n练习: 多步预测")
    
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 500)
    data = np.sin(t)
    
    seq_len = 30
    pred_len = 10
    
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
    y = torch.FloatTensor(np.array(y))
    
    print(f"输入形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    
    class MultiStepPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_len):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_len)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    model = MultiStepPredictor(input_size=1, hidden_size=64, output_len=pred_len)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练模型...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    model.eval()
    with torch.no_grad():
        sample_input = X[0:1]
        prediction = model(sample_input)
    
    print(f"\n预测结果形状: {prediction.shape}")
    print(f"真实值: {y[0].numpy()}")
    print(f"预测值: {prediction[0].numpy()}")

def autoregressive_prediction():
    print("\n" + "=" * 50)
    print("自回归预测练习")
    print("=" * 50)
    
    print("\n练习: 自回归生成序列")
    
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 300)
    data = np.sin(t)
    
    seq_len = 20
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
    
    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
    
    model = NextStepPredictor(input_size=1, hidden_size=32, num_layers=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("训练模型...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            
            x = X[i:i+1, :-1]
            y_true = X[i:i+1, 1:]
            
            y_pred = []
            hidden = None
            for t in range(x.shape[1]):
                out, hidden = model.lstm(x[:, t:t+1], hidden)
                pred = model.fc(out)
                y_pred.append(pred)
            
            y_pred = torch.cat(y_pred, dim=1)
            loss = criterion(y_pred.squeeze(-1), y_true.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(X):.6f}")
    
    print("\n自回归生成...")
    model.eval()
    with torch.no_grad():
        seed = X[0:1, :10]
        generated = seed.clone()
        hidden = None
        
        for _ in range(100):
            out, hidden = model.lstm(generated[:, -1:], hidden)
            pred = model.fc(out)
            generated = torch.cat([generated, pred.unsqueeze(1)], dim=1)
    
    plt.figure(figsize=(12, 4))
    plt.plot(generated[0].numpy(), label='生成序列')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.title('自回归生成序列')
    plt.legend()
    plt.grid(True)
    plt.savefig('autoregressive.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    test_sequence_prediction()
    test_next_step_prediction()
    test_multi_step_prediction()
    autoregressive_prediction()
    print("\n" + "=" * 50)
    print("序列预测练习完成!")
    print("=" * 50)
