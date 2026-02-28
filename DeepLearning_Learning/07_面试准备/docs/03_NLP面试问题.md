# NLP面试问题

## 1. 词向量

### 1.1 Word2Vec

**Q: 解释Word2Vec的两种模型**

```
CBOW (Continuous Bag of Words):
- 输入: 上下文词
- 输出: 中心词
- 适合小数据集，训练快

Skip-gram:
- 输入: 中心词
- 输出: 上下文词
- 适合大数据集，对稀有词效果好

训练技巧:
1. 负采样 (Negative Sampling)
   - 将多分类转为二分类
   - 随机采样负样本
   - 计算效率高

2. 层次Softmax
   - 使用霍夫曼树
   - 将复杂度从O(V)降到O(logV)

3. 高频词下采样
   - 减少高频词的训练次数
   - 提高稀有词的学习效果
```

### 1.2 GloVe

**Q: GloVe与Word2Vec的区别？**

```
Word2Vec:
- 基于局部上下文窗口
- 预测任务驱动
- 隐式利用共现信息

GloVe:
- 基于全局词共现矩阵
- 显式利用统计信息
- 目标函数: 
  J = Σ f(X_ij)(w_i·w̃_j + b_i + b̃_j - log X_ij)²

GloVe优势:
- 结合全局统计和局部上下文
- 训练效率高
- 在词类比任务上表现好
```

## 2. 语言模型

### 2.1 N-gram

**Q: N-gram语言模型的优缺点？**

```
优点:
- 简单高效
- 可解释性强
- 不需要训练神经网络

缺点:
- 稀疏性问题
- 无法捕获长距离依赖
- 需要大量存储空间

平滑技术:
1. Laplace平滑: P(w) = (c(w) + 1) / (N + V)
2. Good-Turing: 用未见词频率估计
3. Kneser-Ney: 最有效的平滑方法
4. 插值平滑: 结合不同n-gram

困惑度:
- PPL = exp(-1/N Σ log P(w_i|context))
- 越低越好
- 表示模型对下一个词的平均分支因子
```

### 2.2 神经语言模型

**Q: 神经语言模型相比N-gram的优势？**

```
优势:
1. 解决稀疏性问题
   - 词嵌入表示
   - 泛化到未见过的情况

2. 捕获长距离依赖
   - RNN/LSTM/Transformer
   - 理论上无限上下文

3. 参数共享
   - 词嵌入共享
   - 减少参数量

4. 迁移学习
   - 预训练+微调
   - 少样本学习

架构演进:
- 前馈NNLM: 固定窗口
- RNNLM: 递归结构
- Transformer LM: 并行计算
```

## 3. 序列标注

### 3.1 HMM vs CRF

**Q: HMM和CRF的区别？**

```
HMM (隐马尔可夫模型):
- 生成模型
- 建模P(X,Y) = P(Y)P(X|Y)
- 马尔可夫假设
- 输出独立性假设

CRF (条件随机场):
- 判别模型
- 建模P(Y|X)
- 可以使用任意特征
- 全局最优

CRF优势:
- 不需要独立性假设
- 可以利用丰富的特征
- 全局归一化，避免标签偏置

BiLSTM-CRF:
- BiLSTM提取特征
- CRF学习标签转移
- 结合深度学习和传统方法
```

### 3.2 命名实体识别

**Q: NER的常用方法和评估指标？**

```
方法:
1. 规则方法
   - 正则表达式
   - 词典匹配

2. 传统机器学习
   - CRF
   - HMM

3. 深度学习
   - BiLSTM-CRF
   - BERT-CRF
   - Biaffine NER

评估指标:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2PR / (P + R)

实体级评估:
- 严格匹配: 边界和类型都正确
- 宽松匹配: 只看类型
```

## 4. 文本分类

**Q: 文本分类的常用方法？**

```
传统方法:
1. 朴素贝叶斯
   - P(c|d) ∝ P(c)ΠP(w_i|c)
   - 简单高效

2. SVM
   - 高维稀疏特征
   - 核方法处理非线性

3. 逻辑回归
   - 可解释性强
   - 输出概率

深度学习方法:
1. TextCNN
   - 多通道卷积
   - 捕获局部特征

2. BiLSTM
   - 捕获序列信息
   - 双向上下文

3. BERT
   - 预训练表示
   - 迁移学习
   - SOTA效果
```

## 5. 机器翻译

### 5.1 Seq2Seq

**Q: Seq2Seq模型的关键组件？**

```
编码器:
- 将源序列编码为固定长度向量
- 通常使用双向RNN
- 输出上下文向量

解码器:
- 根据上下文向量生成目标序列
- 自回归生成
- Teacher Forcing训练

注意力机制:
- 解决固定长度瓶颈
- 动态关注源序列不同位置
- 对齐源语言和目标语言

训练技巧:
1. Teacher Forcing
2. Scheduled Sampling
3. 梯度裁剪
4. 标签平滑
```

### 5.2 评估指标

**Q: BLEU指标的原理？**

```
BLEU (Bilingual Evaluation Understudy):

计算方法:
1. 计算n-gram匹配
   - 通常计算1-4 gram

2. 计算精确率
   p_n = 匹配的n-gram数 / 候选翻译n-gram总数

3. 几何平均
   BP × exp(Σ w_n log p_n)

BP (Brevity Penalty):
- 惩罚过短的翻译
- BP = exp(1 - r/c)，如果c<r

局限性:
- 不考虑语义
- 需要多个参考翻译
- 对词形变化敏感
```

## 6. 预训练模型

### 6.1 BERT

**Q: BERT的预训练任务？**

```
1. MLM (Masked Language Model)
   - 随机遮蔽15%的token
   - 80%替换为[MASK]
   - 10%替换为随机词
   - 10%保持不变
   - 预测被遮蔽的词

2. NSP (Next Sentence Prediction)
   - 输入两个句子
   - 预测是否相邻
   - 50%正样本，50%负样本

输入表示:
- Token Embeddings
- Segment Embeddings
- Position Embeddings

微调:
- 文本分类: [CLS]输出接分类层
- 序列标注: 每个token输出接分类层
- 问答: 预测答案起止位置
```

### 6.2 GPT

**Q: GPT与BERT的区别？**

```
架构:
- BERT: Transformer编码器
- GPT: Transformer解码器

注意力:
- BERT: 双向注意力
- GPT: 单向(因果)注意力

预训练:
- BERT: MLM + NSP
- GPT: 自回归语言建模

应用:
- BERT: 理解任务(分类、标注)
- GPT: 生成任务(文本生成)

生成能力:
- BERT: 不擅长生成
- GPT: 强大的生成能力
```

## 7. 注意力机制

**Q: 注意力机制的变体？**

```
1. 加性注意力 (Bahdanau)
   score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)
   - 计算复杂度较高

2. 点积注意力 (Luong)
   score(h_t, h_s) = h_t^T h_s
   - 计算高效
   - 需要维度匹配

3. 缩放点积注意力
   score = h_t^T h_s / √d_k
   - 防止点积过大
   - Transformer使用

4. 多头注意力
   - 多个子空间并行
   - 捕获不同依赖关系

5. 自注意力
   - Q=K=V
   - 序列内部注意力
```

## 8. 文本表示

**Q: TF-IDF的计算方法？**

```
TF (Term Frequency):
- TF(t,d) = 词t在文档d中的出现次数 / 文档d的总词数

IDF (Inverse Document Frequency):
- IDF(t) = log(文档总数 / 包含词t的文档数)

TF-IDF:
- TF-IDF(t,d) = TF(t,d) × IDF(t)

意义:
- TF: 词在文档中的重要性
- IDF: 词的区分度
- 高TF-IDF: 词在文档中重要且具有区分度

改进:
1. 平滑IDF: log((N+1)/(n+1)) + 1
2. 子线性TF: 1 + log(tf)
3. 归一化: L2归一化
```
