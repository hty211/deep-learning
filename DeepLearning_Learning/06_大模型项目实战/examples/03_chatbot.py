import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import deque

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

class ChatMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
    
    def add_message(self, role: str, content: str):
        self.history.append({'role': role, 'content': content})
    
    def get_context(self, max_tokens: int = 512) -> str:
        context_parts = []
        total_length = 0
        
        for msg in reversed(self.history):
            msg_text = f"{msg['role']}: {msg['content']}"
            if total_length + len(msg_text) > max_tokens:
                break
            context_parts.insert(0, msg_text)
            total_length += len(msg_text)
        
        return '\n'.join(context_parts)
    
    def clear(self):
        self.history.clear()

class SimpleChatModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, 
                 num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50,
                 temperature: float = 1.0, top_k: int = 10) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(generated)
                next_token_logits = outputs[0, -1, :] / temperature
                
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[next_token_idx]
                
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return generated

class ChatBot:
    def __init__(self, model: SimpleChatModel, tokenizer: SimpleTokenizer,
                 system_prompt: str = "你是一个友好的AI助手。"):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.memory = ChatMemory(max_turns=10)
    
    def chat(self, user_input: str) -> str:
        self.memory.add_message('user', user_input)
        
        context = self.memory.get_context()
        
        full_prompt = f"{self.system_prompt}\n\n{context}\nassistant:"
        
        input_ids = torch.tensor([self.tokenizer.encode(full_prompt)])
        
        output_ids = self.model.generate(input_ids, max_length=30, temperature=0.8)
        
        response = self.tokenizer.decode(output_ids[0].tolist())
        
        self.memory.add_message('assistant', response)
        
        return response
    
    def reset(self):
        self.memory.clear()
        print("对话已重置")

class DialogueManager:
    def __init__(self, chatbot: ChatBot):
        self.chatbot = chatbot
        self.commands = {
            '/help': self.show_help,
            '/reset': self.reset_conversation,
            '/history': self.show_history
        }
    
    def show_help(self):
        print("\n可用命令:")
        print("  /help - 显示帮助")
        print("  /reset - 重置对话")
        print("  /history - 显示对话历史")
        print("  /quit - 退出程序")
    
    def reset_conversation(self):
        self.chatbot.reset()
    
    def show_history(self):
        print("\n对话历史:")
        for msg in self.chatbot.memory.history:
            print(f"  {msg['role']}: {msg['content']}")
    
    def process_input(self, user_input: str) -> bool:
        if user_input.startswith('/'):
            command = user_input.split()[0]
            if command == '/quit':
                return False
            elif command in self.commands:
                self.commands[command]()
            else:
                print(f"未知命令: {command}")
        else:
            response = self.chatbot.chat(user_input)
            print(f"\n助手: {response}")
        
        return True
    
    def run(self):
        print("=" * 60)
        print("智能对话助手")
        print("输入 /help 查看帮助，/quit 退出")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if not user_input:
                    continue
                
                if not self.process_input(user_input):
                    break
                    
            except KeyboardInterrupt:
                print("\n\n再见!")
                break

def create_vocab():
    vocab = {
        '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
        '你': 4, '好': 5, '是': 6, '我': 7, '的': 8,
        '什么': 9, '怎么': 10, '吗': 11, '？': 12,
        '助': 13, '手': 14, '可': 15, '以': 16,
        '帮': 17, '忙': 18, '谢': 19, '谢': 20,
        '问': 21, '答': 22, '人': 23, '工': 24,
        '智': 25, '能': 26, '机': 27, '器': 28,
        '学': 29, '习': 30, '深': 31, '度': 32,
        '神': 33, '经': 34, '网': 35, '络': 36,
        '自': 37, '然': 38, '语': 39, '言': 40,
        '处': 41, '理': 42, '欢': 43, '迎': 44
    }
    return vocab

def main():
    print("=" * 60)
    print("智能对话助手示例")
    print("=" * 60)
    
    vocab = create_vocab()
    tokenizer = SimpleTokenizer(vocab)
    
    model = SimpleChatModel(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_heads=4
    )
    
    chatbot = ChatBot(
        model=model,
        tokenizer=tokenizer,
        system_prompt="你是一个友好的AI助手，乐于帮助用户解决问题。"
    )
    
    dialogue_manager = DialogueManager(chatbot)
    
    print("\n演示对话:")
    print("-" * 40)
    
    demo_inputs = [
        "你好",
        "什么是人工智能",
        "谢谢"
    ]
    
    for user_input in demo_inputs:
        print(f"\n用户: {user_input}")
        response = chatbot.chat(user_input)
        print(f"助手: {response}")
    
    print("\n" + "=" * 60)
    print("对话助手示例完成!")
    print("=" * 60)
    
    print("\n提示: 在实际应用中，可以运行 dialogue_manager.run() 进入交互模式")

if __name__ == "__main__":
    main()
