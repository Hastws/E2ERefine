import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 生成数据
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 绘制半月形数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Class 1")
plt.title("make_moons Dataset (noise=0.1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
