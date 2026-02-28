import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = text.lower().split()[:self.max_len]
        
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        
        return torch.tensor(indices), torch.tensor(label)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, 
                 num_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, _ = self.lstm(embedded)
        
        out = self.fc(lstm_out[:, -1, :])
        return out

def create_sample_data():
    positive_texts = [
        "this movie is great and amazing",
        "i really loved this film",
        "excellent performance by the actors",
        "wonderful story and beautiful scenes",
        "best movie i have ever seen",
        "highly recommend this film",
        "fantastic acting and directing",
        "a masterpiece of cinema",
        "truly enjoyable experience",
        "brilliant and captivating"
    ]
    
    negative_texts = [
        "this movie is terrible and boring",
        "i hated this film so much",
        "worst performance ever seen",
        "awful story and bad scenes",
        "waste of time and money",
        "do not recommend this film",
        "terrible acting and directing",
        "a disaster of cinema",
        "truly painful experience",
        "horrible and disappointing"
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    
    return texts, labels, vocab

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=20, device='cpu'):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs

def plot_results(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('sentiment_training.png', dpi=150)
    plt.show()

def predict_sentiment(model, text, vocab, max_len=50, device='cpu'):
    model.eval()
    
    tokens = text.lower().split()[:max_len]
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    
    input_tensor = torch.tensor([indices]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
    
    sentiment = "正面" if prediction == 1 else "负面"
    confidence = prob[0][prediction].item()
    
    return sentiment, confidence

def main():
    print("=" * 50)
    print("文本情感分析")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n创建示例数据...")
    texts, labels, vocab = create_sample_data()
    print(f"样本数量: {len(texts)}")
    print(f"词汇表大小: {len(vocab)}")
    
    dataset = SentimentDataset(texts, labels, vocab, max_len=20)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    print("\n创建模型...")
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_size=64,
        hidden_size=128,
        output_size=2,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=30, device=device
    )
    
    print("\n绘制训练曲线...")
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    print("\n测试预测...")
    test_texts = [
        "this movie is great and wonderful",
        "terrible and boring film",
        "amazing acting performance",
        "worst movie ever seen"
    ]
    
    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, text, vocab, device=device)
        print(f"文本: '{text}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2%})\n")
    
    print("=" * 50)
    print("情感分析完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
