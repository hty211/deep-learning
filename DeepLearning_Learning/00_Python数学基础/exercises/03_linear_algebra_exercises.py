import numpy as np

def test_vectors():
    print("=" * 50)
    print("向量练习")
    print("=" * 50)
    
    print("\n练习1: 向量运算")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print(f"向量a: {a}")
    print(f"向量b: {b}")
    print(f"加法: {a + b}")
    print(f"点积: {np.dot(a, b)}")
    print(f"叉积: {np.cross(a, b)}")
    
    print("\n练习2: 向量的模")
    v = np.array([3, 4])
    norm = np.linalg.norm(v)
    print(f"向量v: {v}")
    print(f"模: {norm}")
    print(f"单位向量: {v / norm}")
    
    print("\n练习3: 向量夹角")
    a = np.array([1, 0])
    b = np.array([1, 1])
    
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cos_angle)
    print(f"向量a: {a}")
    print(f"向量b: {b}")
    print(f"夹角(弧度): {angle:.4f}")
    print(f"夹角(角度): {np.degrees(angle):.4f}")

def test_matrices():
    print("\n" + "=" * 50)
    print("矩阵练习")
    print("=" * 50)
    
    print("\n练习1: 矩阵运算")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"矩阵A:\n{A}")
    print(f"矩阵B:\n{B}")
    print(f"加法:\n{A + B}")
    print(f"矩阵乘法:\n{A @ B}")
    print(f"转置:\n{A.T}")
    
    print("\n练习2: 特殊矩阵")
    print(f"单位矩阵(3x3):\n{np.eye(3)}")
    print(f"对角矩阵:\n{np.diag([1, 2, 3])}")
    print(f"零矩阵:\n{np.zeros((3, 3))}")
    
    print("\n练习3: 行列式和逆矩阵")
    A = np.array([[1, 2], [3, 4]])
    det = np.linalg.det(A)
    print(f"矩阵A:\n{A}")
    print(f"行列式: {det:.4f}")
    
    if det != 0:
        A_inv = np.linalg.inv(A)
        print(f"逆矩阵:\n{A_inv}")
        print(f"验证 A @ A_inv:\n{A @ A_inv}")

def test_eigen():
    print("\n" + "=" * 50)
    print("特征值与特征向量练习")
    print("=" * 50)
    
    print("\n练习1: 计算特征值和特征向量")
    A = np.array([[4, 2], [1, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"矩阵A:\n{A}")
    print(f"特征值: {eigenvalues}")
    print(f"特征向量:\n{eigenvectors}")
    
    print("\n验证特征值和特征向量:")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        print(f"λ = {lam:.4f}")
        print(f"A @ v = {A @ v}")
        print(f"λ * v = {lam * v}")
        print()

def test_svd():
    print("\n" + "=" * 50)
    print("SVD分解练习")
    print("=" * 50)
    
    print("\n练习1: SVD分解")
    A = np.array([[1, 2, 3], [4, 5, 6]])
    U, S, Vt = np.linalg.svd(A)
    
    print(f"矩阵A:\n{A}")
    print(f"U:\n{U}")
    print(f"奇异值S: {S}")
    print(f"Vt:\n{Vt}")
    
    print("\n重构矩阵:")
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    Sigma[:len(S), :len(S)] = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    print(f"重构结果:\n{A_reconstructed}")
    
    print("\n练习2: 使用SVD进行降维")
    np.random.seed(42)
    A = np.random.rand(10, 5)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    k = 2
    A_reduced = U[:, :k] @ np.diag(S[:k])
    print(f"原始矩阵形状: {A.shape}")
    print(f"降维后形状: {A_reduced.shape}")

def test_linear_equations():
    print("\n" + "=" * 50)
    print("线性方程组练习")
    print("=" * 50)
    
    print("\n练习1: 求解线性方程组")
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 8])
    
    print(f"方程组: Ax = b")
    print(f"A:\n{A}")
    print(f"b: {b}")
    
    x = np.linalg.solve(A, b)
    print(f"解x: {x}")
    print(f"验证 Ax: {A @ x}")
    
    print("\n练习2: 最小二乘解")
    A = np.array([[1, 1], [1, 2], [1, 3]])
    b = np.array([1, 2, 2])
    
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"超定方程组最小二乘解: {x}")
    print(f"残差: {residuals}")

def linear_algebra_challenges():
    print("\n" + "=" * 50)
    print("线性代数挑战题")
    print("=" * 50)
    
    print("\n挑战1: PCA实现")
    def pca(X, n_components):
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        return X_centered @ eigenvectors[:, :n_components]
    
    np.random.seed(42)
    X = np.random.rand(100, 5)
    X_pca = pca(X, 2)
    print(f"原始数据形状: {X.shape}")
    print(f"PCA后形状: {X_pca.shape}")
    
    print("\n挑战2: 矩阵幂")
    def matrix_power(A, n):
        result = np.eye(A.shape[0])
        for _ in range(n):
            result = result @ A
        return result
    
    A = np.array([[1, 1], [0, 1]])
    print(f"矩阵A:\n{A}")
    print(f"A^5:\n{matrix_power(A, 5)}")
    
    print("\n挑战3: 计算矩阵的条件数")
    def condition_number(A):
        U, S, Vt = np.linalg.svd(A)
        return S.max() / S.min()
    
    A = np.array([[1, 2], [3, 4]])
    print(f"矩阵A:\n{A}")
    print(f"条件数: {condition_number(A):.4f}")
    print(f"NumPy计算的条件数: {np.linalg.cond(A):.4f}")

if __name__ == "__main__":
    test_vectors()
    test_matrices()
    test_eigen()
    test_svd()
    test_linear_equations()
    linear_algebra_challenges()
    print("\n" + "=" * 50)
    print("线性代数练习完成!")
    print("=" * 50)
