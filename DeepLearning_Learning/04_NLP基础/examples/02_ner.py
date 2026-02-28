import torch
import torch.nn as nn
import numpy as np

class NERDataset:
    def __init__(self, sentences, tags, word_vocab, tag_vocab):
        self.sentences = sentences
        self.tags = tags
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
    
    def __len__(self):
        return len(self.sentences)
    
    def get_item(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        word_indices = [self.word_vocab.get(w, self.word_vocab['<UNK>']) for w in sentence]
        tag_indices = [self.tag_vocab[t] for t in tags]
        
        return torch.tensor(word_indices), torch.tensor(tag_indices)

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_tags, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        self.fc = nn.Linear(hidden_size * 2, num_tags)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)

def create_ner_data():
    sentences = [
        ["John", "lives", "in", "New", "York"],
        ["Mary", "works", "at", "Google", "in", "California"],
        ["Apple", "is", "based", "in", "Cupertino"]
    ]
    
    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC"],
        ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
        ["B-ORG", "O", "O", "O", "B-LOC"]
    ]
    
    word_vocab = {'<PAD>': 0, '<UNK>': 1}
    tag_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
    
    for sentence in sentences:
        for word in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
    
    return sentences, tags, word_vocab, tag_vocab

def main():
    print("=" * 50)
    print("命名实体识别示例")
    print("=" * 50)
    
    sentences, tags, word_vocab, tag_vocab = create_ner_data()
    print(f"句子数: {len(sentences)}")
    print(f"词汇表大小: {len(word_vocab)}")
    print(f"标签数: {len(tag_vocab)}")
    
    print("\n标签类型:")
    for tag, idx in tag_vocab.items():
        print(f"  {tag}: {idx}")
    
    dataset = NERDataset(sentences, tags, word_vocab, tag_vocab)
    
    model = BiLSTMNER(
        vocab_size=len(word_vocab),
        embed_size=64,
        hidden_size=128,
        num_tags=len(tag_vocab)
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练模型...")
    for epoch in range(20):
        total_loss = 0
        
        for i in range(len(dataset)):
            words, tags = dataset.get_item(i)
            
            optimizer.zero_grad()
            outputs = model(words.unsqueeze(0))
            loss = criterion(outputs.squeeze(0), tags)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
    
    print("\n测试预测...")
    idx_to_tag = {v: k for k, v in tag_vocab.items()}
    model.eval()
    
    test_sentence = ["John", "works", "in", "California"]
    word_indices = [word_vocab.get(w, word_vocab['<UNK>']) for w in test_sentence]
    input_tensor = torch.tensor([word_indices])
    
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs.argmax(2)[0].tolist()
    
    print(f"输入: {test_sentence}")
    print(f"预测: {[idx_to_tag[p] for p in predictions]}")
    
    print("\n" + "=" * 50)
    print("NER示例完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
