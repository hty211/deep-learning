import numpy as np

def test_numpy_basics():
    print("=" * 50)
    print("NumPy基础练习")
    print("=" * 50)
    
    print("\n练习1: 创建数组")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"一维数组: {arr}")
    
    zeros = np.zeros((3, 4))
    print(f"3x4零矩阵:\n{zeros}")
    
    ones = np.ones((2, 3))
    print(f"2x3一矩阵:\n{ones}")
    
    arange_arr = np.arange(0, 10, 2)
    print(f"等差数组(0-10, 步长2): {arange_arr}")
    
    linspace_arr = np.linspace(0, 1, 5)
    print(f"等分数组(0-1, 5个数): {linspace_arr}")
    
    print("\n练习2: 数组运算")
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"np.sqrt(a) = {np.sqrt(a)}")
    
    print("\n练习3: 数组索引和切片")
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"原数组:\n{arr}")
    print(f"第一行: {arr[0, :]}")
    print(f"第一列: {arr[:, 0]}")
    print(f"前两行两列:\n{arr[:2, :2]}")
    
    print("\n练习4: 布尔索引")
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"原数组: {arr}")
    print(f"大于5的元素: {arr[arr > 5]}")
    arr[arr > 5] = 0
    print(f"大于5的元素替换为0: {arr}")
    
    print("\n练习5: 统计运算")
    arr = np.random.rand(3, 4)
    print(f"随机数组:\n{arr}")
    print(f"求和: {np.sum(arr):.4f}")
    print(f"均值: {np.mean(arr):.4f}")
    print(f"标准差: {np.std(arr):.4f}")
    print(f"按列求和: {np.sum(arr, axis=0)}")
    print(f"按行求和: {np.sum(arr, axis=1)}")
    
    print("\n练习6: 矩阵运算")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"矩阵A:\n{A}")
    print(f"矩阵B:\n{B}")
    print(f"矩阵乘法 A @ B:\n{A @ B}")
    print(f"转置 A.T:\n{A.T}")
    print(f"行列式 det(A): {np.linalg.det(A):.4f}")
    print(f"逆矩阵 inv(A):\n{np.linalg.inv(A)}")
    
    print("\n练习7: 形状操作")
    arr = np.arange(12)
    print(f"原数组: {arr}")
    reshaped = arr.reshape(3, 4)
    print(f"重塑为3x4:\n{reshaped}")
    print(f"展平: {reshaped.flatten()}")
    
    print("\n练习8: 广播机制")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([10, 20, 30])
    print(f"矩阵a:\n{a}")
    print(f"向量b: {b}")
    print(f"a + b (广播):\n{a + b}")

def numpy_challenges():
    print("\n" + "=" * 50)
    print("NumPy挑战题")
    print("=" * 50)
    
    print("\n挑战1: 实现softmax函数")
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    x = np.array([2.0, 1.0, 0.1])
    print(f"输入: {x}")
    print(f"Softmax输出: {softmax(x)}")
    print(f"验证和为1: {softmax(x).sum():.6f}")
    
    print("\n挑战2: 实现one-hot编码")
    def one_hot(labels, num_classes):
        one_hot_matrix = np.zeros((len(labels), num_classes))
        one_hot_matrix[np.arange(len(labels)), labels] = 1
        return one_hot_matrix
    
    labels = np.array([0, 2, 1, 3])
    print(f"标签: {labels}")
    print(f"One-hot编码:\n{one_hot(labels, 4)}")
    
    print("\n挑战3: 计算欧氏距离矩阵")
    def euclidean_distance_matrix(X):
        sum_sq = np.sum(X**2, axis=1)
        dist = np.sqrt(sum_sq[:, np.newaxis] + sum_sq - 2 * X @ X.T)
        return dist
    
    X = np.random.rand(4, 3)
    print(f"数据点:\n{X}")
    print(f"距离矩阵:\n{euclidean_distance_matrix(X)}")
    
    print("\n挑战4: 实现min-max标准化")
    def min_max_normalize(X):
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    print(f"原始数据:\n{X}")
    print(f"标准化后:\n{min_max_normalize(X)}")
    
    print("\n挑战5: 实现Z-score标准化")
    def z_score_normalize(X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    print(f"原始数据:\n{X}")
    print(f"Z-score标准化后:\n{z_score_normalize(X)}")

if __name__ == "__main__":
    test_numpy_basics()
    numpy_challenges()
    print("\n" + "=" * 50)
    print("NumPy练习完成!")
    print("=" * 50)
