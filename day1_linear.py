# linear.py
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置项
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)

    def __str__(self):
        return f"LinearRegression(weights={self.weights}, bias={self.bias})"   


if __name__ == "__main__":
    # 生成样本数据
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # 创建并训练模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    predictions = model.predict(X)

    # 输出模型参数
    print(model)

    # 输出均方误差
    score = model.score(X, y)
    print(f"Mean Squared Error: {score}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(X, predictions, color="red", linewidth=2, label="Prediction Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
