# NumPy基础教程

## 1. NumPy简介

NumPy是Python科学计算的基础库，提供了高性能的多维数组对象和相关工具。

```python
import numpy as np
```

## 2. 创建数组

### 2.1 从列表创建

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)
print(type(a))
print(a.dtype)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
print(b.shape)
```

### 2.2 使用内置函数创建

```python
zeros = np.zeros((3, 4))
print(zeros)

ones = np.ones((2, 3, 4))
print(ones)

empty = np.empty((2, 3))
print(empty)

arange_arr = np.arange(0, 10, 2)
print(arange_arr)

linspace_arr = np.linspace(0, 1, 5)
print(linspace_arr)
```

### 2.3 随机数组

```python
rand_arr = np.random.rand(3, 4)
print(rand_arr)

randn_arr = np.random.randn(3, 4)
print(randn_arr)

randint_arr = np.random.randint(0, 10, (3, 4))
print(randint_arr)

np.random.seed(42)
```

## 3. 数组属性

```python
arr = np.random.rand(3, 4, 5)

print(f"维度: {arr.ndim}")
print(f"形状: {arr.shape}")
print(f"元素总数: {arr.size}")
print(f"数据类型: {arr.dtype}")
print(f"每个元素大小: {arr.itemsize} bytes")
```

## 4. 数组索引和切片

### 4.1 基本索引

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr[0, 0])
print(arr[1, 2])
print(arr[-1, -1])
```

### 4.2 切片

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(arr[0, :])
print(arr[:, 0])
print(arr[0:2, 1:3])
print(arr[::2, ::2])
```

### 4.3 花式索引

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

indices = [0, 2, 4, 6]
print(arr[indices])

arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[[0, 2], [1, 2]])
```

### 4.4 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

mask = arr > 5
print(mask)
print(arr[mask])

arr[arr > 5] = 0
print(arr)
```

## 5. 数组运算

### 5.1 算术运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(np.sqrt(a))
print(np.exp(a))
print(np.log(a))
```

### 5.2 统计运算

```python
arr = np.random.rand(3, 4)

print(f"求和: {np.sum(arr)}")
print(f"均值: {np.mean(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"方差: {np.var(arr)}")
print(f"最小值: {np.min(arr)}")
print(f"最大值: {np.max(arr)}")
print(f"最小值索引: {np.argmin(arr)}")
print(f"最大值索引: {np.argmax(arr)}")

print(f"按列求和: {np.sum(arr, axis=0)}")
print(f"按行求和: {np.sum(arr, axis=1)}")
```

## 6. 数组形状操作

### 6.1 reshape

```python
arr = np.arange(12)

reshaped = arr.reshape(3, 4)
print(reshaped)

reshaped = arr.reshape(3, -1)
print(reshaped)

flattened = reshaped.flatten()
print(flattened)

raveled = reshaped.ravel()
print(raveled)
```

### 6.2 转置

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.T)
print(np.transpose(arr))
```

### 6.3 合并数组

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

print(np.concatenate([a, b], axis=0))

b = np.array([[5], [6]])
print(np.concatenate([a, b], axis=1))

print(np.vstack([a, b.reshape(1, 2)]))
print(np.hstack([a, b]))
```

### 6.4 分割数组

```python
arr = np.arange(12).reshape(3, 4)

print(np.split(arr, 2, axis=1))
print(np.split(arr, 3, axis=0))
```

## 7. 广播机制

广播允许不同形状的数组进行运算：

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

print(a + b)

c = np.array([[10], [20]])
print(a + c)
```

广播规则：
1. 从后向前比较形状
2. 维度相等或其中一个为1
3. 缺失的维度视为1

## 8. 矩阵运算

### 8.1 矩阵乘法

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))
print(A @ B)
print(np.matmul(A, B))
```

### 8.2 线性代数运算

```python
A = np.array([[1, 2], [3, 4]])

print(f"行列式: {np.linalg.det(A)}")
print(f"逆矩阵:\n{np.linalg.inv(A)}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

b = np.array([5, 6])
x = np.linalg.solve(A, b)
print(f"线性方程组解: {x}")
```

## 9. 实用技巧

### 9.1 条件判断

```python
arr = np.array([1, 2, 3, 4, 5, 6])

result = np.where(arr > 3, arr, 0)
print(result)

print(np.clip(arr, 2, 5))
```

### 9.2 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(np.sort(arr))
print(np.argsort(arr))

arr_2d = np.array([[3, 1, 4], [1, 5, 9]])
print(np.sort(arr_2d, axis=1))
```

### 9.3 去重

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4])

print(np.unique(arr))
print(np.unique(arr, return_counts=True))
```

## 10. 性能优化

```python
import time

size = 1000000

a = list(range(size))
b = list(range(size))

start = time.time()
c = [x + y for x, y in zip(a, b)]
print(f"列表运算时间: {time.time() - start:.4f}s")

a_np = np.arange(size)
b_np = np.arange(size)

start = time.time()
c_np = a_np + b_np
print(f"NumPy运算时间: {time.time() - start:.4f}s")
```

## 练习

完成 [NumPy练习](../exercises/01_numpy_exercises.py) 来巩固所学知识。
