import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 读取原始数据
file_path = r"D:\dev\Python\DepressionForecast\Student Depression Dataset.csv"
df = pd.read_csv(file_path)

# 2. 填补缺失值（Financial Stress）
df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].mode()[0])

# 3. 将 'Yes' / 'No' 字段映射为数值（0 / 1）
binary_map = {"Yes": 1, "No": 0}
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(binary_map)
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(binary_map)

# 4. 删除无关列 'id'
df.drop(columns=['id'], inplace=True)

# 5. 独热编码（One-Hot Encoding）多类别分类变量
categorical_columns = [
    'Gender', 'City', 'Profession', 'Sleep Duration',
    'Dietary Habits', 'Degree'
]
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 6. 标准化数值特征（除目标变量 'Depression'）
features = df_encoded.drop(columns=['Depression'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 7. 构建清洗后的 DataFrame
df_cleaned = pd.DataFrame(X_scaled, columns=features.columns)
df_cleaned['Depression'] = df_encoded['Depression'].values

# 8. 保存清洗后的数据
output_path = r"D:\dev\Python\DepressionForecast\Cleaned_Depression_Dataset.csv"
df_cleaned.to_csv(output_path, index=False)

print("✅ 数据清洗完成，已保存至：", output_path)
