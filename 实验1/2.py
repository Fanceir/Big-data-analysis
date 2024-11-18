import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

# 加载加州住房数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 用于存储结果
mse_list = []
r2_list = []

# 调试不同的 alpha 值
alphas = np.logspace(-2, 2, 10)  # 使用对数间隔的 alpha 值
for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # 存储结果
    mse_list.append(mse_ridge)
    r2_list.append(r2_ridge)

    print(f"Alpha: {alpha}")
    print("Mean Squared Error:", mse_ridge)
    print("R^2 Score:", r2_ridge)
    print("-" * 30)

# 绘制 MSE 和 R² 的变化图
plt.figure(figsize=(12, 6))

# MSE 图
plt.subplot(1, 2, 1)
plt.plot(alphas, mse_list, marker="o", label="MSE", color="blue")
plt.xscale("log")  # 使用对数刻度
plt.xlabel("Alpha (log scale)")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Alpha")
plt.grid(True)
plt.legend()

# R² 图
plt.subplot(1, 2, 2)
plt.plot(alphas, r2_list, marker="o", label="R² Score", color="green")
plt.xscale("log")  # 使用对数刻度
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² Score")
plt.title("R² Score vs Alpha")
plt.grid(True)
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig("./ridge_results.png")
