import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math

class TextGenerator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.95, repetition_penalty: float = 1.2) -> str:
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated)
                next_token_logits = outputs[0, -1, :]
                
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
                
                next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_id:
                    break
        
        return self.tokenizer.decode(generated[0].tolist())
    
    def _apply_repetition_penalty(self, logits, generated, penalty):
        if penalty == 1.0:
            return logits
        
        for token_id in generated[0].unique():
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        
        return logits
    
    def _top_k_filtering(self, logits, top_k):
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits

class SimpleGenerativeModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 num_layers: int = 4, num_heads: int = 4,
                 max_position: int = 512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        causal_mask = self._generate_causal_mask(seq_length, input_ids.device)
        
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)
        
        return self.lm_head(x)
    
    def _generate_causal_mask(self, size: int, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class TextGenerationPipeline:
    def __init__(self, model, tokenizer):
        self.generator = TextGenerator(model, tokenizer)
        self.templates = {
            'story': PromptTemplate(
                "写一个关于{topic}的故事:\n",
                ['topic']
            ),
            'summary': PromptTemplate(
                "请总结以下内容:\n{content}\n\n总结:",
                ['content']
            ),
            'translation': PromptTemplate(
                "将以下{source_lang}翻译成{target_lang}:\n{text}\n\n翻译:",
                ['source_lang', 'target_lang', 'text']
            ),
            'qa': PromptTemplate(
                "问题: {question}\n\n回答:",
                ['question']
            ),
            'code': PromptTemplate(
                "请用{language}编写{description}:\n",
                ['language', 'description']
            )
        }
    
    def generate_story(self, topic: str, **kwargs) -> str:
        prompt = self.templates['story'].format(topic=topic)
        return self.generator.generate(prompt, **kwargs)
    
    def summarize(self, content: str, **kwargs) -> str:
        prompt = self.templates['summary'].format(content=content)
        return self.generator.generate(prompt, **kwargs)
    
    def translate(self, text: str, source_lang: str = "中文",
                  target_lang: str = "英文", **kwargs) -> str:
        prompt = self.templates['translation'].format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text
        )
        return self.generator.generate(prompt, **kwargs)
    
    def answer_question(self, question: str, **kwargs) -> str:
        prompt = self.templates['qa'].format(question=question)
        return self.generator.generate(prompt, **kwargs)
    
    def generate_code(self, description: str, language: str = "Python", **kwargs) -> str:
        prompt = self.templates['code'].format(language=language, description=description)
        return self.generator.generate(prompt, **kwargs)

class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get('<PAD>', 0)
        self.unk_id = vocab.get('<UNK>', 1)
        self.bos_id = vocab.get('<BOS>', 2)
        self.eos_id = vocab.get('<EOS>', 3)
    
    def encode(self, text: str) -> List[int]:
        tokens = text.lower().split()
        return [self.vocab.get(t, self.unk_id) for t in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if token not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                    tokens.append(token)
        return ' '.join(tokens)

def create_vocab():
    vocab = {
        '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
        '写': 4, '一': 5, '个': 6, '关': 7, '于': 8,
        '的': 9, '故': 10, '事': 11, '请': 12, '总': 13,
        '结': 14, '翻': 15, '译': 16, '问': 17, '答': 18,
        '人': 19, '工': 20, '智': 21, '能': 22, '机': 23,
        '器': 24, '学': 25, '习': 26, '深': 27, '度': 28,
        '神': 29, '经': 30, '网': 31, '络': 32, '自': 33,
        '然': 34, '语': 35, '言': 36, '处': 37, '理': 38,
        '代': 39, '码': 40, '函': 41, '数': 42, '类': 43,
        'python': 44, 'java': 45, 'javascript': 46
    }
    return vocab

def main():
    print("=" * 60)
    print("文本生成应用示例")
    print("=" * 60)
    
    vocab = create_vocab()
    tokenizer = SimpleTokenizer(vocab)
    
    model = SimpleGenerativeModel(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_heads=4
    )
    
    pipeline = TextGenerationPipeline(model, tokenizer)
    
    print("\n1. 故事生成")
    print("-" * 40)
    story = pipeline.generate_story("人工智能", max_length=20)
    print(f"生成结果: {story}")
    
    print("\n2. 文本摘要")
    print("-" * 40)
    content = "人工智能是计算机科学的一个分支 它致力于创建能够执行需要人类智能的任务的系统"
    summary = pipeline.summarize(content, max_length=15)
    print(f"原文: {content}")
    print(f"摘要: {summary}")
    
    print("\n3. 问答")
    print("-" * 40)
    question = "什么是机器学习"
    answer = pipeline.answer_question(question, max_length=20)
    print(f"问题: {question}")
    print(f"回答: {answer}")
    
    print("\n4. 代码生成")
    print("-" * 40)
    code = pipeline.generate_code("排序函数", "Python", max_length=20)
    print(f"生成代码: {code}")
    
    print("\n" + "=" * 60)
    print("文本生成示例完成!")
    print("=" * 60)
    
    print("\n生成参数说明:")
    print("  - temperature: 控制随机性 (0.1-2.0)")
    print("  - top_k: 保留概率最高的k个词")
    print("  - top_p: 核采样，保留累积概率p")
    print("  - repetition_penalty: 重复惩罚")
    print("  - max_length: 最大生成长度")

if __name__ == "__main__":
    main()
