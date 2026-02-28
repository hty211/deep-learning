import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    def __init__(self, sentences, tags, vocab, tag_vocab, max_length=128):
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab
        self.tag_vocab = tag_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        token_ids = [self.vocab.get('[CLS]', 2)]
        tag_ids = [self.tag_vocab.get('O', 0)]
        
        for word, tag in zip(sentence, tags):
            token_ids.append(self.vocab.get(word, self.vocab.get('<UNK>', 1)))
            tag_ids.append(self.tag_vocab.get(tag, 0))
        
        token_ids.append(self.vocab.get('[SEP]', 3))
        tag_ids.append(self.tag_vocab.get('O', 0))
        
        attention_mask = [1] * len(token_ids)
        
        while len(token_ids) < self.max_length:
            token_ids.append(self.vocab.get('<PAD>', 0))
            tag_ids.append(-100)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(token_ids[:self.max_length]),
            'attention_mask': torch.tensor(attention_mask[:self.max_length]),
            'labels': torch.tensor(tag_ids[:self.max_length])
        }

class BertForTokenClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, 
                 num_labels, max_length=128, dropout=0.1):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        hidden_states = self.encoder(embeddings)
        
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        return logits

def create_ner_data():
    sentences = [
        ["John", "lives", "in", "New", "York", "City"],
        ["Mary", "works", "at", "Google", "in", "California"],
        ["Apple", "Inc", "is", "based", "in", "Cupertino"],
        ["Microsoft", "was", "founded", "by", "Bill", "Gates"]
    ]
    
    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC"],
        ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
        ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC"],
        ["B-ORG", "O", "O", "O", "B-PER", "I-PER"]
    ]
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
    tag_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
    
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return sentences, tags, vocab, tag_vocab

def main():
    print("=" * 60)
    print("BERT命名实体识别示例")
    print("=" * 60)
    
    sentences, tags, vocab, tag_vocab = create_ner_data()
    
    print(f"\n数据集:")
    print(f"  句子数: {len(sentences)}")
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  标签数: {len(tag_vocab)}")
    
    print("\n标签类型:")
    for tag, idx in sorted(tag_vocab.items(), key=lambda x: x[1]):
        print(f"  {idx}: {tag}")
    
    dataset = NERDataset(sentences, tags, vocab, tag_vocab, max_length=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = BertForTokenClassification(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_labels=len(tag_vocab)
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print("\n训练模型...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            loss = criterion(outputs.view(-1, model.num_labels), batch['labels'].view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    print("\n测试预测...")
    model.eval()
    
    idx_to_tag = {v: k for k, v in tag_vocab.items()}
    
    test_sentence = ["John", "works", "at", "Apple", "in", "California"]
    token_ids = [vocab.get('[CLS]', 2)]
    for word in test_sentence:
        token_ids.append(vocab.get(word, vocab.get('<UNK>', 1)))
    token_ids.append(vocab.get('[SEP]', 3))
    
    while len(token_ids) < 32:
        token_ids.append(0)
    
    input_tensor = torch.tensor([token_ids])
    
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs.argmax(2)[0].tolist()
    
    print(f"\n输入: {test_sentence}")
    print("预测:")
    for i, (word, pred) in enumerate(zip(test_sentence, predictions[1:len(test_sentence)+1])):
        tag = idx_to_tag.get(pred, 'O')
        print(f"  {word}: {tag}")
    
    print("\n" + "=" * 60)
    print("BERT NER示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
