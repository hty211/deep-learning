import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextClassificationDataset(Dataset):
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

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

def create_sample_data():
    texts = [
        "this movie is great and amazing",
        "i really loved this film excellent",
        "wonderful story and beautiful scenes",
        "best movie i have ever seen fantastic",
        "this movie is terrible and boring",
        "i hated this film so much awful",
        "worst performance ever seen bad",
        "waste of time and money horrible"
    ]
    
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    
    return texts, labels, vocab

def main():
    print("=" * 50)
    print("文本分类示例")
    print("=" * 50)
    
    texts, labels, vocab = create_sample_data()
    print(f"样本数: {len(texts)}")
    print(f"词汇表大小: {len(vocab)}")
    
    dataset = TextClassificationDataset(texts, labels, vocab, max_len=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_size=64,
        hidden_size=128,
        num_classes=2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练模型...")
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for texts_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Acc: {100.*correct/total:.2f}%")
    
    print("\n测试预测...")
    test_texts = ["this movie is great", "terrible and boring film"]
    model.eval()
    
    for text in test_texts:
        tokens = text.lower().split()
        indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
        indices += [0] * (20 - len(indices))
        input_tensor = torch.tensor([indices])
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
        
        sentiment = "正面" if pred == 1 else "负面"
        print(f"  '{text}' -> {sentiment}")
    
    print("\n" + "=" * 50)
    print("文本分类完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
