# Matplotlib基础教程

## 1. Matplotlib简介

Matplotlib是Python最常用的绑图库，用于创建静态、动态和交互式可视化。

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

## 2. 基础绑图

### 2.1 折线图

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('正弦函数')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```

### 2.2 多条线

```python
x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='blue', linestyle='-')
plt.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
plt.title('三角函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.3 线条样式

```python
x = np.linspace(0, 10, 50)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, x, 'r-')
plt.title('实线')

plt.subplot(2, 2, 2)
plt.plot(x, x, 'g--')
plt.title('虚线')

plt.subplot(2, 2, 3)
plt.plot(x, x, 'b:')
plt.title('点线')

plt.subplot(2, 2, 4)
plt.plot(x, x, 'm-.')
plt.title('点划线')

plt.tight_layout()
plt.show()
```

## 3. 常用图表类型

### 3.1 散点图

```python
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = np.random.rand(50) * 100

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(label='颜色值')
plt.title('散点图')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 3.2 柱状图

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue', edgecolor='navy')
plt.title('柱状图')
plt.xlabel('类别')
plt.ylabel('数值')

for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha='center')

plt.show()
```

### 3.3 水平柱状图

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.barh(categories, values, color='lightcoral')
plt.title('水平柱状图')
plt.xlabel('数值')
plt.ylabel('类别')
plt.show()
```

### 3.4 直方图

```python
np.random.seed(42)
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='teal', edgecolor='white', alpha=0.7)
plt.title('直方图 - 正态分布')
plt.xlabel('值')
plt.ylabel('频数')
plt.axvline(data.mean(), color='red', linestyle='--', label=f'均值: {data.mean():.2f}')
plt.legend()
plt.show()
```

### 3.5 饼图

```python
labels = ['A', 'B', 'C', 'D']
sizes = [30, 25, 25, 20]
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0, 0)

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('饼图')
plt.axis('equal')
plt.show()
```

### 3.6 箱线图

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['A', 'B', 'C'])
plt.title('箱线图')
plt.ylabel('值')
plt.grid(True, axis='y')
plt.show()
```

## 4. 子图

### 4.1 subplot

```python
x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, np.sin(x))
plt.title('sin(x)')

plt.subplot(2, 2, 2)
plt.plot(x, np.cos(x))
plt.title('cos(x)')

plt.subplot(2, 2, 3)
plt.plot(x, np.tan(x))
plt.title('tan(x)')
plt.ylim(-5, 5)

plt.subplot(2, 2, 4)
plt.plot(x, np.exp(x))
plt.title('exp(x)')

plt.tight_layout()
plt.show()
```

### 4.2 subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('tan(x)')
axes[1, 0].set_ylim(-5, 5)

axes[1, 1].plot(x, np.exp(x))
axes[1, 1].set_title('exp(x)')

plt.tight_layout()
plt.show()
```

## 5. 样式设置

### 5.1 颜色映射

```python
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
plt.colorbar(label='Z值')
plt.title('等高线图')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 5.2 使用样式

```python
print(plt.style.available)

plt.style.use('seaborn-v0_8')

x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x))
plt.title('使用seaborn样式')
plt.show()
```

## 6. 3D绑图

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D曲面图')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
```

## 7. 保存图片

```python
x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x))
plt.title('正弦函数')

plt.savefig('sin_function.png', dpi=300, bbox_inches='tight')
plt.savefig('sin_function.pdf', bbox_inches='tight')

print("图片已保存")
```

## 8. 实用技巧

### 8.1 添加注释

```python
x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x))
plt.title('正弦函数')

plt.annotate('最大值', xy=(np.pi/2, 1), xytext=(np.pi/2 + 0.5, 1.2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=12, color='red')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)

plt.show()
```

### 8.2 双Y轴

```python
x = np.linspace(0, 10, 100)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)', color=color)
ax1.plot(x, np.sin(x), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('exp(x)', color=color)
ax2.plot(x, np.exp(x), color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('双Y轴图')
plt.show()
```

## 9. 数据可视化最佳实践

1. **选择合适的图表类型**
   - 比较：柱状图
   - 趋势：折线图
   - 分布：直方图、箱线图
   - 关系：散点图
   - 占比：饼图

2. **保持简洁**
   - 避免过多颜色
   - 去除不必要的装饰
   - 使用清晰的标签

3. **突出重点**
   - 使用颜色、大小强调关键信息
   - 添加注释说明重要点

## 练习

完成 [数据分析小项目](../examples/01_data_analysis_project.py) 来巩固所学知识。
