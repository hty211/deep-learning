# RAG技术

## 1. RAG概述

```python
def explain_rag():
    print("RAG (Retrieval-Augmented Generation):")
    print("  结合检索和生成的混合架构")
    print("\n核心思想:")
    print("  1. 检索: 从知识库中检索相关文档")
    print("  2. 增强: 将检索结果作为上下文")
    print("  3. 生成: 基于增强上下文生成回答")

explain_rag()
```

## 2. RAG架构

### 2.1 整体流程

```python
class SimpleRAG:
    def __init__(self, embedding_model, llm, vector_store):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_store = vector_store
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results
    
    def augment(self, query, retrieved_docs):
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        augmented_prompt = f"""
基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文:
{context}

问题: {query}

回答:
"""
        return augmented_prompt
    
    def generate(self, prompt):
        response = self.llm.generate(prompt)
        return response
    
    def query(self, question, top_k=5):
        retrieved = self.retrieve(question, top_k)
        prompt = self.augment(question, retrieved)
        answer = self.generate(prompt)
        return answer, retrieved

print("RAG流程:")
print("  1. 用户提问")
print("  2. 问题向量化")
print("  3. 向量检索")
print("  4. 构建增强提示")
print("  5. LLM生成回答")
```

## 3. 向量数据库

### 3.1 简单向量存储

```python
import numpy as np

class SimpleVectorStore:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.vectors = []
        self.documents = []
    
    def add(self, documents, embeddings):
        for doc, emb in zip(documents, embeddings):
            self.documents.append(doc)
            self.vectors.append(emb)
        
        self.vectors = np.array(self.vectors)
    
    def search(self, query_embedding, top_k=5):
        if len(self.vectors) == 0:
            return []
        
        similarities = np.dot(self.vectors, query_embedding) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results

print("\n向量存储功能:")
print("  - 存储文档和向量")
print("  - 相似度搜索")
print("  - 支持增量添加")
```

### 3.2 常用向量数据库

```python
def vector_db_comparison():
    print("\n向量数据库对比:")
    print("=" * 80)
    print(f"{'数据库':<15} {'特点':<30} {'适用场景':<20}")
    print("-" * 80)
    print(f"{'FAISS':<15} {'Facebook开源，高效':<30} {'本地部署':<20}")
    print(f"{'Chroma':<15} {'轻量级，易用':<30} {'快速原型':<20}")
    print(f"{'Pinecone':<15} {'云托管，可扩展':<30} {'生产环境':<20}")
    print(f"{'Milvus':<15} {'高性能，分布式':<30} {'大规模应用':<20}")
    print(f"{'Weaviate':<15} {'语义搜索，GraphQL':<30} {'知识图谱':<20}")
    print("=" * 80)

vector_db_comparison()
```

## 4. 文档处理

### 4.1 文档分块

```python
class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                split_point = max(last_period, last_newline)
                
                if split_point > start + self.chunk_size // 2:
                    chunk = text[start:start + split_point + 1]
                    end = start + split_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def split_documents(self, documents):
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'content': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i
                    }
                })
        return all_chunks

print("\n文档分块策略:")
print("  1. 固定大小分块")
print("  2. 句子分块")
print("  3. 段落分块")
print("  4. 语义分块")
print("  5. 递归分块")
```

### 4.2 文档加载

```python
class DocumentLoader:
    @staticmethod
    def load_text(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{'content': content, 'metadata': {'source': file_path}}]
    
    @staticmethod
    def load_pdf(file_path):
        print(f"加载PDF: {file_path}")
        print("  需要安装: pip install pypdf")
        return []
    
    @staticmethod
    def load_json(file_path):
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

print("\n支持的文档格式:")
print("  - TXT: 纯文本")
print("  - PDF: 需要pypdf库")
print("  - JSON: 结构化数据")
print("  - CSV: 表格数据")
print("  - Markdown: 文档格式")
```

## 5. 嵌入模型

### 5.1 嵌入模型选择

```python
def embedding_model_comparison():
    print("\n嵌入模型对比:")
    print("=" * 80)
    print(f"{'模型':<25} {'维度':<10} {'特点':<30}")
    print("-" * 80)
    print(f"{'text-embedding-ada-002':<25} {'1536':<10} {'OpenAI，高质量':<30}")
    print(f"{'bge-large-zh':<25} {'1024':<10} {'中文优化':<30}")
    print(f"{'m3e-base':<25} {'768':<10} {'多语言':<30}")
    print(f"{'all-MiniLM-L6-v2':<25} {'384':<10} {'轻量级':<30}")
    print("=" * 80)

embedding_model_comparison()
```

### 5.2 简单嵌入模型

```python
class SimpleEmbedding:
    def __init__(self, vocab, embedding_dim=128):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
    
    def encode(self, text):
        tokens = text.lower().split()
        
        vectors = []
        for token in tokens:
            if token in self.vocab:
                vectors.append(self.embeddings[self.vocab[token]])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts):
        return np.array([self.encode(text) for text in texts])

print("\n嵌入模型功能:")
print("  - 文本向量化")
print("  - 语义相似度计算")
print("  - 批量处理")
```

## 6. 高级RAG技术

### 6.1 混合检索

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_index, alpha=0.5):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha = alpha
    
    def search(self, query, query_embedding, top_k=5):
        vector_results = self.vector_store.search(query_embedding, top_k * 2)
        bm25_results = self.bm25_index.search(query, top_k * 2)
        
        combined_scores = {}
        
        for i, result in enumerate(vector_results):
            doc_id = result['id']
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + \
                self.alpha * (1 / (i + 1))
        
        for i, result in enumerate(bm25_results):
            doc_id = result['id']
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + \
                (1 - self.alpha) * (1 / (i + 1))
        
        sorted_results = sorted(combined_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]

print("\n混合检索优势:")
print("  - 结合语义和关键词匹配")
print("  - 提高召回率")
print("  - 更鲁棒")
```

### 6.2 重排序

```python
class Reranker:
    def __init__(self, model):
        self.model = model
    
    def rerank(self, query, documents, top_k=5):
        scores = []
        for doc in documents:
            score = self.model.score(query, doc['content'])
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scores[:top_k]]

print("\n重排序技术:")
print("  - 对初步检索结果重新排序")
print("  - 使用更精确的模型")
print("  - 提高最终结果质量")
```

## 7. RAG优化技巧

```python
def rag_optimization_tips():
    print("\nRAG优化技巧:")
    print("\n1. 检索优化:")
    print("   - 使用混合检索")
    print("   - 调整chunk大小")
    print("   - 增加重排序")
    
    print("\n2. 生成优化:")
    print("   - 优化提示模板")
    print("   - 控制上下文长度")
    print("   - 添加引用来源")
    
    print("\n3. 知识库优化:")
    print("   - 定期更新知识")
    print("   - 去重和清洗")
    print("   - 添加元数据")
    
    print("\n4. 评估指标:")
    print("   - 检索准确率")
    print("   - 回答相关性")
    print("   - 事实准确性")

rag_optimization_tips()
```

## 8. 总结

| 组件 | 功能 | 关键技术 |
|------|------|---------|
| 文档处理 | 分块、加载 | TextSplitter |
| 嵌入模型 | 向量化 | BGE, M3E |
| 向量存储 | 存储、检索 | FAISS, Chroma |
| 检索器 | 相似度搜索 | 混合检索 |
| 重排序 | 结果优化 | Cross-Encoder |
| 生成器 | 回答生成 | LLM |
