import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    print("=" * 50)
    print("数据分析小项目: 泰坦尼克号数据分析")
    print("=" * 50)
    
    np.random.seed(42)
    n = 891
    
    data = {
        'PassengerId': range(1, n + 1),
        'Survived': np.random.binomial(1, 0.38, n),
        'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(1, n + 1)],
        'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 14, n).clip(0.5, 80),
        'SibSp': np.random.choice(range(6), n, p=[0.68, 0.23, 0.06, 0.02, 0.006, 0.004]),
        'Parch': np.random.choice(range(7), n, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.003, 0.002]),
        'Fare': np.random.exponential(32, n).clip(0, 512),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09])
    }
    
    mask = np.random.random(n) < 0.2
    data['Age'][mask] = np.nan
    
    df = pd.DataFrame(data)
    
    print("\n数据预览:")
    print(df.head())
    
    print("\n数据信息:")
    print(df.info())
    
    print("\n描述统计:")
    print(df.describe())
    
    return df

def data_cleaning(df):
    print("\n" + "=" * 50)
    print("数据清洗")
    print("=" * 50)
    
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    df_clean = df.copy()
    
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    
    df_clean.dropna(subset=['Embarked'], inplace=True)
    
    print(f"\n清洗后缺失值统计:")
    print(df_clean.isnull().sum())
    
    return df_clean

def exploratory_analysis(df):
    print("\n" + "=" * 50)
    print("探索性数据分析")
    print("=" * 50)
    
    print("\n1. 存活率分析")
    survival_rate = df['Survived'].mean()
    print(f"总体存活率: {survival_rate:.2%}")
    
    print("\n各舱位存活率:")
    pclass_survival = df.groupby('Pclass')['Survived'].mean()
    print(pclass_survival)
    
    print("\n各性别存活率:")
    sex_survival = df.groupby('Sex')['Survived'].mean()
    print(sex_survival)
    
    print("\n2. 年龄分布分析")
    print(f"平均年龄: {df['Age'].mean():.2f}")
    print(f"年龄中位数: {df['Age'].median():.2f}")
    print(f"年龄标准差: {df['Age'].std():.2f}")
    
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], 
                            labels=['儿童', '青年', '中年', '老年'])
    
    print("\n各年龄段存活率:")
    age_survival = df.groupby('AgeGroup')['Survived'].mean()
    print(age_survival)
    
    return df

def visualization(df):
    print("\n" + "=" * 50)
    print("数据可视化")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax1 = axes[0, 0]
    survival_counts = df['Survived'].value_counts()
    ax1.bar(['死亡', '存活'], survival_counts.values, color=['red', 'green'])
    ax1.set_title('存活情况分布')
    ax1.set_ylabel('人数')
    for i, v in enumerate(survival_counts.values):
        ax1.text(i, v + 10, str(v), ha='center')
    
    ax2 = axes[0, 1]
    pclass_survival = df.groupby('Pclass')['Survived'].mean()
    ax2.bar(['一等舱', '二等舱', '三等舱'], pclass_survival.values, 
            color=['gold', 'silver', 'brown'])
    ax2.set_title('各舱位存活率')
    ax2.set_ylabel('存活率')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(pclass_survival.values):
        ax2.text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    ax3 = axes[0, 2]
    sex_survival = df.groupby('Sex')['Survived'].mean()
    ax3.bar(sex_survival.index, sex_survival.values, color=['pink', 'blue'])
    ax3.set_title('各性别存活率')
    ax3.set_ylabel('存活率')
    ax3.set_ylim(0, 1)
    for i, v in enumerate(sex_survival.values):
        ax3.text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    ax4 = axes[1, 0]
    ax4.hist(df['Age'], bins=30, color='skyblue', edgecolor='white')
    ax4.axvline(df['Age'].mean(), color='red', linestyle='--', label=f'均值: {df["Age"].mean():.1f}')
    ax4.set_title('年龄分布')
    ax4.set_xlabel('年龄')
    ax4.set_ylabel('人数')
    ax4.legend()
    
    ax5 = axes[1, 1]
    df.boxplot(column='Fare', by='Pclass', ax=ax5)
    ax5.set_title('各舱位票价分布')
    ax5.set_xlabel('舱位等级')
    ax5.set_ylabel('票价')
    plt.suptitle('')
    
    ax6 = axes[1, 2]
    age_survival = df.groupby('AgeGroup')['Survived'].mean()
    ax6.bar(age_survival.index, age_survival.values, color='teal')
    ax6.set_title('各年龄段存活率')
    ax6.set_ylabel('存活率')
    ax6.set_ylim(0, 1)
    for i, v in enumerate(age_survival.values):
        ax6.text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('titanic_analysis.png', dpi=150, bbox_inches='tight')
    print("图表已保存为 titanic_analysis.png")
    plt.show()

def feature_engineering(df):
    print("\n" + "=" * 50)
    print("特征工程")
    print("=" * 50)
    
    df_fe = df.copy()
    
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    
    df_fe['Sex_encoded'] = (df_fe['Sex'] == 'male').astype(int)
    
    embarked_dummies = pd.get_dummies(df_fe['Embarked'], prefix='Embarked')
    df_fe = pd.concat([df_fe, embarked_dummies], axis=1)
    
    df_fe['FarePerPerson'] = df_fe['Fare'] / df_fe['FamilySize']
    
    print("新增特征:")
    print(df_fe[['FamilySize', 'IsAlone', 'Sex_encoded', 'FarePerPerson']].head())
    
    print("\n特征与存活率的相关性:")
    numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                        'FamilySize', 'IsAlone', 'Sex_encoded', 'FarePerPerson']
    correlation = df_fe[numeric_features].corr()['Survived'].sort_values(ascending=False)
    print(correlation)
    
    return df_fe

def simple_prediction(df):
    print("\n" + "=" * 50)
    print("简单预测模型")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    features = ['Pclass', 'Sex_encoded', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    X = df[features].fillna(df[features].median())
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2%}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['死亡', '存活']))
    
    print("\n特征重要性:")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

def main():
    df = load_and_explore_data()
    df_clean = data_cleaning(df)
    df_explored = exploratory_analysis(df_clean)
    visualization(df_explored)
    df_fe = feature_engineering(df_explored)
    simple_prediction(df_fe)
    
    print("\n" + "=" * 50)
    print("数据分析项目完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
