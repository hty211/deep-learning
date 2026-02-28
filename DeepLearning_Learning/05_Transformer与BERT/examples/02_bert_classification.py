import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_token_id = vocab.get('<PAD>', 0)
        self.unk_token_id = vocab.get('<UNK>', 1)
        self.cls_token_id = vocab.get('[CLS]', 2)
        self.sep_token_id = vocab.get('[SEP]', 3)
    
    def encode(self, text, max_length=128, padding=True, truncation=True):
        tokens = text.lower().split()
        
        token_ids = [self.cls_token_id]
        for token in tokens[:max_length-2]:
            token_ids.append(self.vocab.get(token, self.unk_token_id))
        token_ids.append(self.sep_token_id)
        
        attention_mask = [1] * len(token_ids)
        
        if padding and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            token_ids += [self.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
        
        return {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

class BertForSequenceClassification(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=256, num_layers=4,
                 num_heads=4, num_classes=2, max_length=128, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embeddings = (
            self.embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.token_type_embedding(token_type_ids)
        )
        
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        
        hidden_states = self.encoder(embeddings, src_key_padding_mask=None)
        
        pooled_output = hidden_states[:, 0]
        pooled_output = self.norm(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits

def create_sample_data():
    texts = [
        "this movie is absolutely fantastic and amazing",
        "i really enjoyed watching this excellent film",
        "great story wonderful acting and beautiful scenes",
        "one of the best movies i have ever seen",
        "this movie is terrible and boring waste of time",
        "i hated this awful film so disappointing",
        "worst movie ever made completely unwatchable",
        "do not watch this horrible piece of garbage"
    ]
    
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
    idx = 4
    
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    
    return texts, labels, vocab

def main():
    print("=" * 60)
    print("BERT文本分类示例")
    print("=" * 60)
    
    texts, labels, vocab = create_sample_data()
    print(f"\n数据集:")
    print(f"  样本数: {len(texts)}")
    print(f"  词汇表大小: {len(vocab)}")
    
    tokenizer = SimpleTokenizer(vocab)
    
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = BertForSequenceClassification(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_classes=2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print("\n训练模型...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['input_ids'],
                batch['attention_mask']
            )
            
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch['label'].size(0)
            correct += predicted.eq(batch['label']).sum().item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Acc: {100.*correct/total:.2f}%")
    
    print("\n测试预测...")
    model.eval()
    test_texts = ["this movie is great and amazing", "terrible and boring film"]
    
    for text in test_texts:
        encoding = tokenizer.encode(text, max_length=32)
        
        with torch.no_grad():
            output = model(encoding['input_ids'], encoding['attention_mask'])
            pred = output.argmax(1).item()
        
        sentiment = "正面" if pred == 1 else "负面"
        print(f"  '{text}' -> {sentiment}")
    
    print("\n" + "=" * 60)
    print("BERT文本分类完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
