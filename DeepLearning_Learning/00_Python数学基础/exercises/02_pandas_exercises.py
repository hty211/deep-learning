import pandas as pd
import numpy as np

def test_pandas_basics():
    print("=" * 50)
    print("Pandas基础练习")
    print("=" * 50)
    
    print("\n练习1: 创建DataFrame")
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40],
        'city': ['New York', 'London', 'Paris', 'Tokyo']
    }
    df = pd.DataFrame(data)
    print(df)
    
    print("\n练习2: 查看数据")
    print(f"前2行:\n{df.head(2)}")
    print(f"\n数据信息:")
    print(df.info())
    print(f"\n描述统计:\n{df.describe()}")
    
    print("\n练习3: 选择数据")
    print(f"选择name列:\n{df['name']}")
    print(f"\n选择多列:\n{df[['name', 'age']]}")
    print(f"\n使用loc选择:\n{df.loc[0:1, ['name', 'age']]}")
    print(f"\n使用iloc选择:\n{df.iloc[0:2, 0:2]}")
    
    print("\n练习4: 条件筛选")
    print(f"年龄大于30:\n{df[df['age'] > 30]}")
    print(f"\n城市在列表中:\n{df[df['city'].isin(['Paris', 'Tokyo'])]}")
    
    print("\n练习5: 添加和删除列")
    df['salary'] = [50000, 60000, 70000, 80000]
    print(f"添加salary列:\n{df}")
    
    df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 35 else 'Senior')
    print(f"\n添加age_group列:\n{df}")
    
    df_dropped = df.drop('age_group', axis=1)
    print(f"\n删除age_group列:\n{df_dropped}")

def test_missing_values():
    print("\n" + "=" * 50)
    print("缺失值处理练习")
    print("=" * 50)
    
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    })
    
    print(f"原始数据:\n{df}")
    print(f"\n检测缺失值:\n{df.isnull()}")
    print(f"\n缺失值计数:\n{df.isnull().sum()}")
    
    print(f"\n删除含缺失值的行:\n{df.dropna()}")
    print(f"\n用0填充缺失值:\n{df.fillna(0)}")
    print(f"\n用均值填充缺失值:\n{df.fillna(df.mean())}")

def test_groupby():
    print("\n" + "=" * 50)
    print("分组聚合练习")
    print("=" * 50)
    
    df = pd.DataFrame({
        'department': ['HR', 'IT', 'HR', 'IT', 'Finance', 'Finance'],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'salary': [50000, 60000, 55000, 70000, 65000, 75000],
        'age': [25, 30, 28, 35, 32, 40]
    })
    
    print(f"原始数据:\n{df}")
    
    print(f"\n按部门分组求薪资均值:\n{df.groupby('department')['salary'].mean()}")
    
    print(f"\n多聚合函数:\n{df.groupby('department')['salary'].agg(['mean', 'sum', 'count'])}")
    
    print(f"\n多列聚合:\n{df.groupby('department').agg({'salary': 'mean', 'age': 'max'})}")

def test_merge():
    print("\n" + "=" * 50)
    print("数据合并练习")
    print("=" * 50)
    
    df1 = pd.DataFrame({
        'employee': ['Alice', 'Bob', 'Charlie'],
        'department': ['HR', 'IT', 'Finance']
    })
    
    df2 = pd.DataFrame({
        'employee': ['Alice', 'Bob', 'David'],
        'salary': [50000, 60000, 70000]
    })
    
    print(f"df1:\n{df1}")
    print(f"\ndf2:\n{df2}")
    
    print(f"\n内连接:\n{pd.merge(df1, df2, on='employee', how='inner')}")
    print(f"\n外连接:\n{pd.merge(df1, df2, on='employee', how='outer')}")
    print(f"\n左连接:\n{pd.merge(df1, df2, on='employee', how='left')}")

def pandas_challenges():
    print("\n" + "=" * 50)
    print("Pandas挑战题")
    print("=" * 50)
    
    print("\n挑战1: 泰坦尼克数据分析")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'passenger_id': range(1, n + 1),
        'survived': np.random.binomial(1, 0.38, n),
        'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n),
        'age': np.random.normal(30, 15, n).clip(0, 80),
        'fare': np.random.exponential(30, n)
    })
    
    print(f"数据预览:\n{df.head()}")
    
    print(f"\n存活率: {df['survived'].mean():.2%}")
    print(f"\n各舱位存活率:\n{df.groupby('pclass')['survived'].mean()}")
    print(f"\n男女存活率:\n{df.groupby('sex')['survived'].mean()}")
    
    print("\n挑战2: 时间序列分析")
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    sales = np.random.randint(100, 500, 30)
    df_ts = pd.DataFrame({'date': dates, 'sales': sales})
    df_ts.set_index('date', inplace=True)
    
    print(f"时间序列数据:\n{df_ts.head()}")
    print(f"\n周均值:\n{df_ts.resample('W').mean()}")
    print(f"\n7天移动平均:\n{df_ts.rolling(window=7).mean().head(10)}")
    
    print("\n挑战3: 数据透视表")
    np.random.seed(42)
    df_pivot = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'sales': np.random.randint(100, 1000, 100)
    })
    
    pivot = df_pivot.pivot_table(values='sales', index='product', 
                                  columns='region', aggfunc='mean')
    print(f"数据透视表:\n{pivot}")

if __name__ == "__main__":
    test_pandas_basics()
    test_missing_values()
    test_groupby()
    test_merge()
    pandas_challenges()
    print("\n" + "=" * 50)
    print("Pandas练习完成!")
    print("=" * 50)
