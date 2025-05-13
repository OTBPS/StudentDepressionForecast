import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy.stats import ttest_rel
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# === 1. 读取并清洗数据 ===
file_path = r"D:\dev\Python\DepressionForecast\Student Depression Dataset.csv"
df = pd.read_csv(file_path)

df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].mode()[0])
binary_map = {"Yes": 1, "No": 0}
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(binary_map)
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(binary_map)
df.drop(columns=['id'], inplace=True)

categorical_columns = ['Gender', 'City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

features = df.drop(columns=['Depression'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
y = df['Depression']

# === 2. 划分训练集和测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 定义模型 ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": [],
    "AUC": []
}

roc_curves = {}
xgb_model = None

# === 4. 模型训练与评估 ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n====== {name} Evaluation ======")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC Score: {auc:.4f}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Depression", "Depression"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=600)
    plt.show()

    results["Model"].append(name)
    results["Accuracy"].append(report["accuracy"])
    results["Precision"].append(report["1"]["precision"])
    results["Recall"].append(report["1"]["recall"])
    results["F1-score"].append(report["1"]["f1-score"])
    results["AUC"].append(auc)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)

    if name == "XGBoost":
        xgb_model = model

# === 5. 可视化柱状图和 ROC 曲线 ===
df_results = pd.DataFrame(results).set_index("Model")

# Bar Chart
df_results.plot(kind='bar', figsize=(12, 6), ylim=(0.6, 1.0))
plt.title("Model Evaluation Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("model_evaluation_bar.png", dpi=600)
plt.show()


# ROC Curve
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=600)
plt.show()


# === 6. SHAP 分析（仅对 XGBoost）===
if xgb_model:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

# === 7. 鲁棒性测试（50次 AUC）===
n_runs = 50
auc_scores = {name: [] for name in models}
splitter = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.2, random_state=42)

for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        auc_scores[name].append(auc)

# 输出鲁棒性结果
print("\n====== AUC Robustness Analysis (50 runs) ======")
for name in auc_scores:
    mean_auc = np.mean(auc_scores[name])
    std_auc = np.std(auc_scores[name])
    print(f"{name}: AUC Mean = {mean_auc:.4f}, Std = {std_auc:.4f}")

# === 8. AUC 显著性检验（配对 t 检验）===
print("\n====== AUC Significance Test (paired t-test) ======")
model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        t_stat, p_val = ttest_rel(auc_scores[m1], auc_scores[m2])
        result = "Significant" if p_val < 0.05 else "Not Significant"
        print(f"{m1} vs {m2}: p = {p_val:.4f} → {result}")
# === Robustness Line Plot ===
plt.figure(figsize=(10, 6))
for model in auc_scores:
    plt.plot(auc_scores[model], label=model)
plt.title("AUC Scores over 50 Runs (Robustness Test)")
plt.xlabel("Run")
plt.ylabel("AUC")
plt.ylim(0.88, 0.94)  # 可根据你的数据调整
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("auc_robustness_line.png", dpi=600)
plt.show()

# === AUC Difference Bar Plot ===
mean_diffs = {
    "Logistic vs RF": np.mean(np.array(auc_scores["Logistic Regression"]) - np.array(auc_scores["Random Forest"])),
    "Logistic vs XGB": np.mean(np.array(auc_scores["Logistic Regression"]) - np.array(auc_scores["XGBoost"])),
    "RF vs XGB": np.mean(np.array(auc_scores["Random Forest"]) - np.array(auc_scores["XGBoost"]))
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(mean_diffs.keys()), y=list(mean_diffs.values()))
plt.title("Mean AUC Difference Between Models (50 Runs)")
plt.ylabel("Mean AUC Difference")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("auc_difference_bar.png", dpi=600)
plt.show()




# 1. 模型评估结果
performance_data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.8384, 0.8292, 0.8267],
    "Precision": [0.8505, 0.8416, 0.8442],
    "Recall": [0.8752, 0.8694, 0.8601],
    "F1-score": [0.8627, 0.8552, 0.8521],
    "AUC": [0.9132, 0.9061, 0.9039]
}
df_perf = pd.DataFrame(performance_data)

# 2. 鲁棒性结果
robust_data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "AUC Mean": [0.9206, 0.9140, 0.9118],
    "AUC Std": [0.0035, 0.0037, 0.0036]
}
df_robust = pd.DataFrame(robust_data)

# 3. 显著性检验结果
pval_data = {
    "Model Comparison": ["Logistic vs Random Forest", "Logistic vs XGBoost", "Random Forest vs XGBoost"],
    "p-value": [0.0000, 0.0000, 0.0000],
    "Significance": ["Significant", "Significant", "Significant"]
}
df_pval = pd.DataFrame(pval_data)

# 保存为 Excel（一个文件三个 sheet）
with pd.ExcelWriter("Final_Model_Report.xlsx") as writer:
    df_perf.to_excel(writer, sheet_name="Performance", index=False)
    df_robust.to_excel(writer, sheet_name="Robustness", index=False)
    df_pval.to_excel(writer, sheet_name="Significance Test", index=False)

print("✅ Final report saved as 'Final_Model_Report.xlsx'")
