# Pandas基础教程

## 1. Pandas简介

Pandas是Python数据分析的核心库，提供了DataFrame和Series两种数据结构。

```python
import pandas as pd
import numpy as np
```

## 2. Series

Series是一维带标签的数组。

### 2.1 创建Series

```python
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])
print(s)

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)

data = {'a': 1, 'b': 2, 'c': 3}
s = pd.Series(data)
print(s)
```

### 2.2 Series操作

```python
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

print(s['a'])
print(s[['a', 'c', 'e']])

print(s[1:3])

print(s[s > 2])

print(s * 2)
print(np.sqrt(s))
```

## 3. DataFrame

DataFrame是二维表格数据结构。

### 3.1 创建DataFrame

```python
import pandas as pd
import numpy as np

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['New York', 'London', 'Paris', 'Tokyo']
}
df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
print(df)

arr = np.random.rand(4, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df)
```

### 3.2 查看数据

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'],
    'salary': [50000, 60000, 70000, 80000, 90000]
})

print(df.head(3))
print(df.tail(2))
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)
print(df.index)
print(df.dtypes)
```

## 4. 数据选择

### 4.1 选择列

```python
print(df['name'])
print(df[['name', 'age']])

print(df.name)
```

### 4.2 选择行

```python
print(df.loc[0])
print(df.loc[[0, 2]])
print(df.loc[0:2])

print(df.iloc[0])
print(df.iloc[[0, 2]])
print(df.iloc[0:2])
```

### 4.3 条件选择

```python
print(df[df['age'] > 30])

print(df[(df['age'] > 30) & (df['salary'] > 70000)])

print(df[df['city'].isin(['Paris', 'Tokyo'])])

print(df.query('age > 30 and salary > 70000'))
```

### 4.4 选择特定值

```python
print(df.loc[0, 'name'])
print(df.iloc[0, 0])

print(df.at[0, 'name'])
print(df.iat[0, 0])
```

## 5. 数据操作

### 5.1 添加列

```python
df['department'] = ['HR', 'IT', 'Finance', 'Marketing', 'Sales']

df['bonus'] = df['salary'] * 0.1

df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 35 else 'Senior')
print(df)
```

### 5.2 删除数据

```python
df_dropped = df.drop('bonus', axis=1)
print(df_dropped)

df.drop('bonus', axis=1, inplace=True)

df_dropped = df.drop([0, 1], axis=0)
print(df_dropped)
```

### 5.3 修改数据

```python
df.loc[0, 'salary'] = 55000

df.loc[df['age'] > 35, 'salary'] = df.loc[df['age'] > 35, 'salary'] * 1.1

df['city'] = df['city'].replace('New York', 'NYC')
```

## 6. 缺失值处理

### 6.1 检测缺失值

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

print(df.isnull())
print(df.isnull().sum())
print(df.isnull().any())
```

### 6.2 删除缺失值

```python
print(df.dropna())
print(df.dropna(axis=1))
print(df.dropna(how='all'))
print(df.dropna(thresh=2))
```

### 6.3 填充缺失值

```python
print(df.fillna(0))

print(df.fillna(df.mean()))

print(df.fillna(method='ffill'))
print(df.fillna(method='bfill'))

print(df['A'].fillna(df['A'].mean()))
```

## 7. 数据分组

### 7.1 GroupBy基础

```python
df = pd.DataFrame({
    'department': ['HR', 'IT', 'HR', 'IT', 'Finance', 'Finance'],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'salary': [50000, 60000, 55000, 70000, 65000, 75000],
    'age': [25, 30, 28, 35, 32, 40]
})

grouped = df.groupby('department')

print(grouped['salary'].mean())
print(grouped['salary'].agg(['mean', 'sum', 'count']))
print(grouped.agg({
    'salary': ['mean', 'max'],
    'age': 'mean'
}))
```

### 7.2 多列分组

```python
df['gender'] = ['F', 'M', 'M', 'M', 'F', 'M']

print(df.groupby(['department', 'gender'])['salary'].mean())
```

### 7.3 应用函数

```python
def top_salary(group):
    return group.nlargest(1, 'salary')

print(df.groupby('department').apply(top_salary))
```

## 8. 数据合并

### 8.1 Concat

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

print(pd.concat([df1, df2], axis=0, ignore_index=True))

df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
print(pd.concat([df1, df3], axis=1))
```

### 8.2 Merge

```python
df1 = pd.DataFrame({
    'employee': ['Alice', 'Bob', 'Charlie'],
    'department': ['HR', 'IT', 'Finance']
})

df2 = pd.DataFrame({
    'employee': ['Alice', 'Bob', 'David'],
    'salary': [50000, 60000, 70000]
})

print(pd.merge(df1, df2, on='employee', how='inner'))
print(pd.merge(df1, df2, on='employee', how='outer'))
print(pd.merge(df1, df2, on='employee', how='left'))
print(pd.merge(df1, df2, on='employee', how='right'))
```

### 8.3 Join

```python
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['a', 'b', 'd'])

print(df1.join(df2, how='outer'))
```

## 9. 数据读写

### 9.1 CSV文件

```python
df.to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')
print(df.head())

df = pd.read_csv('data.csv', encoding='utf-8')
```

### 9.2 Excel文件

```python
df.to_excel('data.xlsx', index=False, sheet_name='Sheet1')

df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(df.head())
```

### 9.3 其他格式

```python
df.to_json('data.json')

df = pd.read_json('data.json')

df.to_sql('table_name', con=engine)

df = pd.read_sql('SELECT * FROM table_name', con=engine)
```

## 10. 数据分析实例

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'quantity': np.random.randint(1, 50, 100)
})

print(df.head())
print(df.info())
print(df.describe())

print(df.groupby('product')['sales'].agg(['mean', 'sum', 'count']))

print(df.pivot_table(values='sales', index='product', columns='region', aggfunc='mean'))

df['month'] = df['date'].dt.month
print(df.groupby('month')['sales'].sum())

print(df.sort_values('sales', ascending=False).head(10))
```

## 练习

完成 [Pandas练习](../exercises/02_pandas_exercises.py) 来巩固所学知识。
