import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 参数对比实验
parameters = [
    {"n_estimators": 10, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 5, "max_features": "log2"},
    {"n_estimators": 500, "max_depth": 20, "max_features": "sqrt"},
]

results = []
for param in parameters:
    model = RandomForestClassifier(
        n_estimators=param["n_estimators"],
        max_depth=param["max_depth"],
        max_features=param["max_features"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((param, acc))

# 打印实验结果
for param, acc in results:
    print(f"参数: {param}, 准确率: {acc:.4f}")

# 可视化实验结果
labels = ["1", "2", "3", "4"]
accuracies = [acc for _, acc in results]

plt.figure(figsize=(10, 6))
plt.barh(labels, accuracies, color="skyblue")
plt.xlabel("Accuracy")
plt.ylabel("Parameters")
plt.title("Parameter Tuning Results")
plt.savefig("./results.png")
print("图像已保存为 results.png")
