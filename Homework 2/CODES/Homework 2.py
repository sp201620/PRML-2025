import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 定义生成3D make-moons数据的函数
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 第三维的正弦变化
    # 生成两类数据：C0 和 C1
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

# 生成训练数据（1000个样本，每类500个）和测试数据（500个样本，每类250个）
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 绘制Training Data图像（倾斜视角）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', marker='o')
ax.set_title("Training Data")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=50, azim=70)  # 设置为倾斜视角
plt.show()

# 绘制Test Data图像（倾斜视角）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='viridis', marker='o')
ax.set_title("Test Data (True Labels)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=50, azim=70)  # 设置倾斜视角
plt.show()

# 定义各分类器
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost + Decision Trees": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42),
    "SVM (Poly Kernel)": SVC(kernel='poly', degree=3, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', gamma='scale', random_state=42)
}

# 对每个分类器进行训练、预测并输出Accuracy，同时绘制Test Data分类结果图（倾斜视角）
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', marker='o')
    ax.set_title(f"Test Data Classified by {name}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=50, azim=70)  # 设置倾斜视角
    plt.show()

# 对每个分类器进行训练，同时绘制Test Data分类结果图
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    # 根据预测是否正确决定颜色：正确为绿色，错误为红色
    colors = ['g' if pred==true else 'r' for pred, true in zip(y_pred, y_test)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=colors, marker='o')
    ax.set_title(f"Test Data Classified by {name}\n(Green: Correct, Red: Incorrect)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
