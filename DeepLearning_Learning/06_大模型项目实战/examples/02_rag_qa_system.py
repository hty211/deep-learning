import numpy as np
from typing import List, Dict, Tuple
import json

class SimpleEmbedding:
    def __init__(self, vocab: Dict[str, int], embedding_dim: int = 128):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.embeddings = np.random.randn(len(vocab), embedding_dim) * 0.1
    
    def encode(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vectors = []
        
        for token in tokens:
            if token in self.vocab:
                vectors.append(self.embeddings[self.vocab[token]])
            else:
                vectors.append(np.zeros(self.embedding_dim))
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(text) for text in texts])

class VectorStore:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vectors: List[np.ndarray] = []
        self.documents: List[Dict] = []
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        for doc, emb in zip(documents, embeddings):
            self.documents.append(doc)
            self.vectors.append(emb)
        
        self.vectors = np.array(self.vectors) if self.vectors else np.array([])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if len(self.vectors) == 0:
            return []
        
        similarities = self._cosine_similarity(query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
        norms = np.where(norms == 0, 1e-10, norms)
        return np.dot(self.vectors, query) / norms

class TextSplitter:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = max(chunk.rfind('。'), chunk.rfind('.'))
                if last_period > self.chunk_size // 2:
                    chunk = text[start:start + last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks

class SimpleLLM:
    def __init__(self):
        self.knowledge = {}
    
    def generate(self, prompt: str) -> str:
        if "什么是" in prompt or "什么是" in prompt:
            return self._extract_answer(prompt)
        return "抱歉，我无法回答这个问题。"
    
    def _extract_answer(self, prompt: str) -> str:
        if "上下文:" in prompt:
            parts = prompt.split("上下文:")[1].split("问题:")
            context = parts[0].strip()
            return f"根据文档内容，{context[:100]}..."
        return "未找到相关信息。"

class RAGSystem:
    def __init__(self, embedding_model: SimpleEmbedding, llm: SimpleLLM):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_store = VectorStore(embedding_model.embedding_dim)
        self.text_splitter = TextSplitter()
    
    def add_documents(self, documents: List[Dict]):
        all_chunks = []
        all_texts = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'content': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i
                    }
                })
                all_texts.append(chunk)
        
        embeddings = self.embedding_model.encode_batch(all_texts)
        self.vector_store.add_documents(all_chunks, embeddings)
        
        print(f"已添加 {len(all_chunks)} 个文档块到知识库")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def augment_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        context = "\n\n".join([
            f"[文档{i+1}]\n{doc['document']['content']}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文:
{context}

问题: {query}

回答:"""
        
        return prompt
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        retrieved_docs = self.retrieve(question, top_k)
        
        augmented_prompt = self.augment_prompt(question, retrieved_docs)
        
        answer = self.llm.generate(augmented_prompt)
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs
        }

def create_sample_knowledge_base():
    documents = [
        {
            'content': '机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需进行明确的编程。机器学习算法使用历史数据作为输入来预测新的输出值。',
            'metadata': {'source': 'ml_intro.txt', 'topic': '机器学习'}
        },
        {
            'content': '深度学习是机器学习的一个子领域，它使用人工神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。',
            'metadata': {'source': 'dl_intro.txt', 'topic': '深度学习'}
        },
        {
            'content': '自然语言处理（NLP）是人工智能和语言学的交叉领域，致力于使计算机能够理解、解释和生成人类语言。NLP应用包括机器翻译、情感分析和问答系统。',
            'metadata': {'source': 'nlp_intro.txt', 'topic': 'NLP'}
        },
        {
            'content': 'Transformer是一种基于自注意力机制的神经网络架构，最初用于自然语言处理任务。BERT和GPT都是基于Transformer的预训练模型。',
            'metadata': {'source': 'transformer.txt', 'topic': 'Transformer'}
        },
        {
            'content': '向量数据库是专门用于存储和检索向量嵌入的数据库。它们在RAG系统中用于快速找到与查询最相关的文档片段。',
            'metadata': {'source': 'vector_db.txt', 'topic': '向量数据库'}
        }
    ]
    
    return documents

def main():
    print("=" * 60)
    print("RAG知识库问答系统示例")
    print("=" * 60)
    
    vocab = {}
    sample_docs = create_sample_knowledge_base()
    for doc in sample_docs:
        for word in doc['content'].split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    embedding_model = SimpleEmbedding(vocab, embedding_dim=64)
    llm = SimpleLLM()
    
    rag = RAGSystem(embedding_model, llm)
    
    print("\n构建知识库...")
    rag.add_documents(sample_docs)
    
    print("\n" + "=" * 60)
    print("开始问答")
    print("=" * 60)
    
    questions = [
        "什么是机器学习？",
        "深度学习和机器学习有什么关系？",
        "Transformer是什么？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        
        result = rag.query(question, top_k=2)
        
        print(f"\n回答: {result['answer']}")
        
        print("\n参考来源:")
        for i, source in enumerate(result['sources']):
            score = source['score']
            content = source['document']['content'][:50]
            print(f"  [{i+1}] 相似度: {score:.4f}")
            print(f"      内容: {content}...")
    
    print("\n" + "=" * 60)
    print("RAG问答系统示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
