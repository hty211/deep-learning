import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def test_probability_basics():
    print("=" * 50)
    print("概率基础练习")
    print("=" * 50)
    
    print("\n练习1: 模拟掷骰子")
    np.random.seed(42)
    rolls = np.random.randint(1, 7, 10000)
    
    print(f"掷骰子10000次")
    for i in range(1, 7):
        count = np.sum(rolls == i)
        print(f"点数{i}: {count}次, 概率估计: {count/len(rolls):.4f}")
    
    print("\n练习2: 条件概率")
    np.random.seed(42)
    n = 1000
    disease = np.random.binomial(1, 0.01, n)
    test = np.zeros(n)
    
    test[disease == 1] = np.random.binomial(1, 0.99, np.sum(disease == 1))
    test[disease == 0] = np.random.binomial(1, 0.05, np.sum(disease == 0))
    
    p_disease = disease.mean()
    p_positive = test.mean()
    p_disease_given_positive = disease[test == 1].mean()
    
    print(f"患病率: {p_disease:.4f}")
    print(f"检测阳性率: {p_positive:.4f}")
    print(f"阳性时患病的概率: {p_disease_given_positive:.4f}")

def test_distributions():
    print("\n" + "=" * 50)
    print("概率分布练习")
    print("=" * 50)
    
    print("\n练习1: 二项分布")
    n, p = 10, 0.5
    samples = np.random.binomial(n, p, 10000)
    
    print(f"二项分布 B({n}, {p})")
    print(f"理论均值: {n*p}")
    print(f"样本均值: {samples.mean():.4f}")
    print(f"理论方差: {n*p*(1-p)}")
    print(f"样本方差: {samples.var():.4f}")
    
    print("\n练习2: 正态分布")
    mu, sigma = 0, 1
    samples = np.random.normal(mu, sigma, 10000)
    
    print(f"正态分布 N({mu}, {sigma}²)")
    print(f"理论均值: {mu}")
    print(f"样本均值: {samples.mean():.4f}")
    print(f"理论标准差: {sigma}")
    print(f"样本标准差: {samples.std():.4f}")
    
    print("\n练习3: 泊松分布")
    lam = 5
    samples = np.random.poisson(lam, 10000)
    
    print(f"泊松分布 Poisson({lam})")
    print(f"理论均值: {lam}")
    print(f"样本均值: {samples.mean():.4f}")
    print(f"理论方差: {lam}")
    print(f"样本方差: {samples.var():.4f}")

def test_expectation_variance():
    print("\n" + "=" * 50)
    print("期望与方差练习")
    print("=" * 50)
    
    print("\n练习1: 计算期望和方差")
    x = np.array([1, 2, 3, 4, 5])
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    
    E_X = np.sum(x * p)
    E_X2 = np.sum(x**2 * p)
    Var_X = E_X2 - E_X**2
    
    print(f"随机变量X: {x}")
    print(f"概率P: {p}")
    print(f"期望E[X]: {E_X:.4f}")
    print(f"方差Var(X): {Var_X:.4f}")
    print(f"标准差: {np.sqrt(Var_X):.4f}")
    
    print("\n练习2: 样本统计量")
    np.random.seed(42)
    samples = np.random.normal(10, 2, 1000)
    
    print(f"样本均值: {samples.mean():.4f}")
    print(f"样本方差: {samples.var():.4f}")
    print(f"样本标准差: {samples.std():.4f}")
    print(f"偏度: {stats.skew(samples):.4f}")
    print(f"峰度: {stats.kurtosis(samples):.4f}")

def test_clt():
    print("\n" + "=" * 50)
    print("中心极限定理练习")
    print("=" * 50)
    
    print("\n练习: 验证中心极限定理")
    np.random.seed(42)
    
    population = np.random.exponential(2, 100000)
    print(f"总体均值: {population.mean():.4f}")
    print(f"总体标准差: {population.std():.4f}")
    
    sample_size = 30
    n_samples = 1000
    sample_means = [np.random.choice(population, sample_size).mean() 
                    for _ in range(n_samples)]
    
    print(f"\n样本大小: {sample_size}")
    print(f"样本数: {n_samples}")
    print(f"样本均值的均值: {np.mean(sample_means):.4f}")
    print(f"样本均值的标准差: {np.std(sample_means):.4f}")
    print(f"理论标准误: {population.std()/np.sqrt(sample_size):.4f}")

def test_bayes():
    print("\n" + "=" * 50)
    print("贝叶斯定理练习")
    print("=" * 50)
    
    print("\n练习: 垃圾邮件分类")
    p_spam = 0.3
    p_word_given_spam = 0.8
    p_word_given_ham = 0.1
    
    p_word = p_word_given_spam * p_spam + p_word_given_ham * (1 - p_spam)
    p_spam_given_word = (p_word_given_spam * p_spam) / p_word
    
    print(f"P(垃圾邮件) = {p_spam}")
    print(f"P(包含某词|垃圾邮件) = {p_word_given_spam}")
    print(f"P(包含某词|正常邮件) = {p_word_given_ham}")
    print(f"P(包含某词) = {p_word:.4f}")
    print(f"P(垃圾邮件|包含某词) = {p_spam_given_word:.4f}")

def probability_challenges():
    print("\n" + "=" * 50)
    print("概率论挑战题")
    print("=" * 50)
    
    print("\n挑战1: 蒙特卡洛估计π")
    np.random.seed(42)
    n = 100000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    
    inside = np.sum(x**2 + y**2 <= 1)
    pi_estimate = 4 * inside / n
    
    print(f"样本数: {n}")
    print(f"π的估计值: {pi_estimate:.6f}")
    print(f"真实值: {np.pi:.6f}")
    print(f"误差: {abs(pi_estimate - np.pi):.6f}")
    
    print("\n挑战2: 生日问题")
    def birthday_probability(n_people):
        p_no_match = 1.0
        for i in range(n_people):
            p_no_match *= (365 - i) / 365
        return 1 - p_no_match
    
    for n in [10, 20, 23, 30, 50]:
        p = birthday_probability(n)
        print(f"{n}人中至少两人同生日的概率: {p:.4f}")
    
    print("\n挑战3: 最大似然估计")
    np.random.seed(42)
    true_mu = 5
    true_sigma = 2
    data = np.random.normal(true_mu, true_sigma, 1000)
    
    mu_mle = data.mean()
    sigma_mle = data.std()
    
    print(f"真实参数: μ={true_mu}, σ={true_sigma}")
    print(f"MLE估计: μ={mu_mle:.4f}, σ={sigma_mle:.4f}")

if __name__ == "__main__":
    test_probability_basics()
    test_distributions()
    test_expectation_variance()
    test_clt()
    test_bayes()
    probability_challenges()
    print("\n" + "=" * 50)
    print("概率论练习完成!")
    print("=" * 50)
