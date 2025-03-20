import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file, assuming the first sheet is training data and the second sheet is testing data
data_file = "Data4Regression.xlsx"
train_df = pd.read_excel(data_file, sheet_name=0)
test_df = pd.read_excel(data_file, sheet_name=1)

# Assume the first column is x and the second column is y
x_train = train_df.iloc[:, 0].values
y_train = train_df.iloc[:, 1].values
x_test = test_df.iloc[:, 0].values
y_test = test_df.iloc[:, 1].values

# Construct the design matrix (adding the intercept term)
X_train = np.column_stack((np.ones(len(x_train)), x_train))
X_test = np.column_stack((np.ones(len(x_test)), x_test))

def mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

### 1. Least Squares (Analytical Solution)
theta_ls = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
y_train_pred_ls = X_train @ theta_ls
y_test_pred_ls = X_test @ theta_ls
mse_train_ls = mse(y_train, y_train_pred_ls)
mse_test_ls = mse(y_test, y_test_pred_ls)

print("[Least Squares]")
print("Training MSE:", mse_train_ls)
print("Testing MSE:", mse_test_ls)

### 2. Gradient Descent Method
def gradient_descent(X, y, learning_rate=0.01, num_iters=1000):
    m, n = X.shape
    theta = np.zeros(n)
    mse_history = []
    for i in range(num_iters):
        grad = (1/m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * grad
        mse_history.append(mse(y, X @ theta))
    return theta, mse_history

theta_gd, mse_history_gd = gradient_descent(X_train, y_train, learning_rate=0.01, num_iters=1000)
y_train_pred_gd = X_train @ theta_gd
y_test_pred_gd = X_test @ theta_gd
mse_train_gd = mse(y_train, y_train_pred_gd)
mse_test_gd = mse(y_test, y_test_pred_gd)

print("\n[Gradient Descent]")
print("Training MSE:", mse_train_gd)
print("Testing MSE:", mse_test_gd)
print("Gradient Descent converged in {} iterations, final training MSE: {:.4f}".format(len(mse_history_gd), mse_history_gd[-1]))

### 3. Newton's Method
def newton_method(X, y, num_iters=10):
    m, n = X.shape
    theta = np.zeros(n)
    mse_history = []
    H = (1/m) * (X.T @ X)  # Hessian (constant matrix for linear regression)
    H_inv = np.linalg.inv(H)
    for i in range(num_iters):
        grad = (1/m) * (X.T @ (X @ theta - y))
        theta = theta - H_inv @ grad
        mse_history.append(mse(y, X @ theta))
    return theta, mse_history

theta_newton, mse_history_newton = newton_method(X_train, y_train, num_iters=10)
y_train_pred_newton = X_train @ theta_newton
y_test_pred_newton = X_test @ theta_newton
mse_train_newton = mse(y_train, y_train_pred_newton)
mse_test_newton = mse(y_test, y_test_pred_newton)

print("\n[Newton's Method]")
print("Training MSE:", mse_train_newton)
print("Testing MSE:", mse_test_newton)
print("Newton's Method converged in {} iterations, final training MSE: {:.4f}".format(len(mse_history_newton), mse_history_newton[-1]))

# Plot training data, testing data, and regression curves from three methods
plt.figure(figsize=(8, 6))
# Plot training and testing data
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Testing Data')

# Create a smooth range of x values for plotting the regression lines
x_range = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 300)
# Calculate the predicted y values for each method
y_range_ls = theta_ls[0] + theta_ls[1] * x_range
y_range_gd = theta_gd[0] + theta_gd[1] * x_range
y_range_newton = theta_newton[0] + theta_newton[1] * x_range

# Plot the regression lines
plt.plot(x_range, y_range_ls, color='red', label='Least Squares')
plt.plot(x_range, y_range_gd, color='orange', linestyle='--', label='Gradient Descent')
plt.plot(x_range, y_range_newton, color='purple', linestyle='-.', label="Newton's Method")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression Curves and Data Points")
plt.legend()
plt.show()

# Plot convergence curves for Gradient Descent and Newton's Method
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mse_history_gd, label="Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient Descent Convergence")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mse_history_newton, marker='o', label="Newton's Method")
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Newton's Method Convergence")
plt.legend()
plt.tight_layout()
plt.show()
