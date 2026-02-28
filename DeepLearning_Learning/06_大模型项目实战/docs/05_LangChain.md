# LangChain框架

## 1. LangChain概述

```python
def explain_langchain():
    print("LangChain:")
    print("  用于构建大语言模型应用的开发框架")
    print("\n核心组件:")
    print("  1. Models: 模型接口")
    print("  2. Prompts: 提示管理")
    print("  3. Memory: 记忆系统")
    print("  4. Chains: 链式调用")
    print("  5. Agents: 智能代理")
    print("  6. Retrievers: 检索器")

explain_langchain()
```

## 2. 模型接口

### 2.1 LLM封装

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: list) -> list:
        pass

class SimpleLLM(BaseLLM):
    def __init__(self, model_name="gpt"):
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        print(f"[{self.model_name}] 生成回答...")
        return f"这是对 '{prompt[:50]}...' 的回答"
    
    def batch_generate(self, prompts: list) -> list:
        return [self.generate(p) for p in prompts]

class ChatLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
    
    def chat(self, messages: list) -> str:
        prompt = self._format_messages(messages)
        return self.model.generate(prompt)
    
    def _format_messages(self, messages):
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"{role}: {content}\n"
        return formatted

print("LLM接口:")
print("  - generate: 单次生成")
print("  - batch_generate: 批量生成")
print("  - chat: 对话生成")
```

## 3. 提示模板

### 3.1 基础模板

```python
class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs):
        for var in self.input_variables:
            if var not in kwargs:
                raise ValueError(f"缺少变量: {var}")
        
        return self.template.format(**kwargs)
    
    def partial(self, **kwargs):
        new_template = self.template.format(**kwargs)
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        return PromptTemplate(new_template, remaining_vars)

template = PromptTemplate(
    template="请用{style}的风格回答问题: {question}",
    input_variables=["style", "question"]
)

result = template.format(style="专业", question="什么是机器学习?")
print(f"\n提示模板示例:\n{result}")

professional_template = template.partial(style="专业")
print(f"\n部分填充:\n{professional_template.format(question='什么是深度学习?')}")
```

### 3.2 对话模板

```python
class ChatPromptTemplate:
    def __init__(self):
        self.messages = []
    
    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})
        return self
    
    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        return self
    
    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
        return self
    
    def format(self, **kwargs):
        formatted_messages = []
        for msg in self.messages:
            content = msg["content"].format(**kwargs) if kwargs else msg["content"]
            formatted_messages.append({"role": msg["role"], "content": content})
        return formatted_messages

chat_template = ChatPromptTemplate() \
    .add_system_message("你是一个{role}。") \
    .add_user_message("{question}")

print("\n对话模板示例:")
print(chat_template.format(role="AI助手", question="你好"))
```

## 4. 记忆系统

### 4.1 对话记忆

```python
class ConversationMemory:
    def __init__(self, max_length=10):
        self.messages = []
        self.max_length = max_length
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        
        if len(self.messages) > self.max_length * 2:
            self.messages = self.messages[-self.max_length * 2:]
    
    def get_messages(self):
        return self.messages.copy()
    
    def clear(self):
        self.messages = []

class BufferWindowMemory:
    def __init__(self, k=5):
        self.k = k
        self.buffer = []
    
    def save_context(self, inputs, outputs):
        self.buffer.append({"input": inputs, "output": outputs})
        if len(self.buffer) > self.k:
            self.buffer.pop(0)
    
    def load_memory_variables(self):
        return {"history": self.buffer}

print("\n记忆类型:")
print("  - ConversationBufferMemory: 保存所有对话")
print("  - ConversationBufferWindowMemory: 保存最近k轮")
print("  - ConversationSummaryMemory: 摘要历史对话")
```

## 5. 链式调用

### 5.1 基础Chain

```python
class Chain:
    def __init__(self, llm, prompt_template, memory=None):
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory = memory
    
    def run(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        response = self.llm.generate(prompt)
        
        if self.memory:
            self.memory.save_context(kwargs, {"response": response})
        
        return response

class SequentialChain:
    def __init__(self, chains, input_variables, output_variables):
        self.chains = chains
        self.input_variables = input_variables
        self.output_variables = output_variables
    
    def run(self, **kwargs):
        results = kwargs.copy()
        
        for chain in self.chains:
            output = chain.run(**results)
            results.update(output if isinstance(output, dict) else {"output": output})
        
        return {k: results[k] for k in self.output_variables if k in results}

print("\nChain类型:")
print("  - LLMChain: 单次LLM调用")
print("  - SequentialChain: 顺序执行")
print("  - SimpleSequentialChain: 简单顺序")
print("  - TransformChain: 数据转换")
```

### 5.2 示例链

```python
class QATranslationChain:
    def __init__(self, llm):
        self.llm = llm
        
        self.qa_prompt = PromptTemplate(
            template="回答以下问题: {question}",
            input_variables=["question"]
        )
        
        self.translate_prompt = PromptTemplate(
            template="将以下内容翻译成{language}: {text}",
            input_variables=["language", "text"]
        )
    
    def run(self, question, language="英文"):
        qa_chain = Chain(self.llm, self.qa_prompt)
        answer = qa_chain.run(question=question)
        
        translate_chain = Chain(self.llm, self.translate_prompt)
        translated = translate_chain.run(language=language, text=answer)
        
        return {"answer": answer, "translation": translated}

print("\n链式调用示例:")
print("  问题 -> 回答 -> 翻译")
```

## 6. 智能代理

### 6.1 Agent基础

```python
class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
    
    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    def plan(self, input_text):
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""
可用工具:
{tool_descriptions}

用户输入: {input_text}

请决定使用哪个工具，以及如何使用。输出格式:
工具: [工具名]
输入: [工具输入]
"""
        return self.llm.generate(prompt)
    
    def execute(self, tool_name, tool_input):
        if tool_name in self.tools:
            return self.tools[tool_name].run(tool_input)
        return f"未知工具: {tool_name}"

def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "计算错误"

def search(query):
    return f"搜索结果: {query}"

tools = [
    Tool("calculator", calculator, "执行数学计算"),
    Tool("search", search, "搜索信息")
]

print("\nAgent组件:")
print("  - Tools: 工具定义")
print("  - AgentExecutor: 执行器")
print("  - Plans: 规划策略")
```

### 6.2 ReAct Agent

```python
class ReActAgent:
    def __init__(self, llm, tools, max_iterations=5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
    
    def run(self, question):
        thought_history = []
        
        for _ in range(self.max_iterations):
            prompt = self._build_prompt(question, thought_history)
            response = self.llm.generate(prompt)
            
            action, action_input = self._parse_response(response)
            
            if action == "Final Answer":
                return action_input
            
            if action in self.tools:
                observation = self.tools[action].run(action_input)
                thought_history.append({
                    "thought": response,
                    "action": action,
                    "observation": observation
                })
        
        return "达到最大迭代次数"
    
    def _build_prompt(self, question, history):
        return f"""
问题: {question}

历史思考:
{self._format_history(history)}

请按以下格式思考:
Thought: [思考过程]
Action: [工具名]
Action Input: [工具输入]
"""
    
    def _parse_response(self, response):
        lines = response.strip().split('\n')
        action = None
        action_input = None
        
        for line in lines:
            if line.startswith("Action:"):
                action = line.split(":", 1)[1].strip()
            elif line.startswith("Action Input:"):
                action_input = line.split(":", 1)[1].strip()
        
        return action, action_input
    
    def _format_history(self, history):
        formatted = []
        for h in history:
            formatted.append(f"Thought: {h['thought']}")
            formatted.append(f"Observation: {h['observation']}")
        return "\n".join(formatted)

print("\nReAct模式:")
print("  Thought -> Action -> Observation -> 循环")
```

## 7. 实用示例

### 7.1 问答系统

```python
class SimpleQA:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationMemory()
        
        self.prompt = PromptTemplate(
            template="根据上下文回答问题:\n上下文: {context}\n问题: {question}\n回答:",
            input_variables=["context", "question"]
        )
    
    def ask(self, question, context=""):
        history = self.memory.get_messages()
        full_context = context + "\n" + self._format_history(history)
        
        prompt = self.prompt.format(context=full_context, question=question)
        answer = self.llm.generate(prompt)
        
        self.memory.add_message("user", question)
        self.memory.add_message("assistant", answer)
        
        return answer
    
    def _format_history(self, messages):
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

print("\n问答系统组件:")
print("  - LLM: 语言模型")
print("  - Memory: 对话记忆")
print("  - Prompt: 提示模板")
```

## 8. 总结

| 组件 | 功能 | 示例 |
|------|------|------|
| Models | 模型接口 | LLM, ChatModel |
| Prompts | 提示管理 | PromptTemplate |
| Memory | 状态管理 | BufferMemory |
| Chains | 流程编排 | SequentialChain |
| Agents | 智能决策 | ReActAgent |
| Tools | 能力扩展 | Calculator, Search |
