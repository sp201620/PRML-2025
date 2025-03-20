import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file (first sheet: training, second sheet: testing)
data_file = "Data4Regression.xlsx"
train_df = pd.read_excel(data_file, sheet_name=0)
test_df = pd.read_excel(data_file, sheet_name=1)

# Extract x and y values (assume first column is x, second column is y)
x_train = train_df.iloc[:, 0].values
y_train = train_df.iloc[:, 1].values
x_test = test_df.iloc[:, 0].values
y_test = test_df.iloc[:, 1].values

# For KNN, we need x as 2D array
X_train = x_train.reshape(-1, 1)

def mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def knn_regression_gaussian(x_query, X_train, y_train, k=5, sigma=1.0):
    """
    KNN regression with Gaussian kernel weighting.
    
    Parameters:
        x_query: array-like, shape (d,) or (1,d), a single query point
        X_train: training data, shape (n_samples, d)
        y_train: training targets, shape (n_samples,)
        k: number of nearest neighbors to use
        sigma: bandwidth parameter for Gaussian kernel
        
    Returns:
        Predicted target value for x_query.
    """
    # Compute Euclidean distances from x_query to each training sample
    distances = np.linalg.norm(X_train - x_query, axis=1)
    # Get indices of the k nearest neighbors
    idx = np.argsort(distances)[:k]
    k_distances = distances[idx]
    k_targets = y_train[idx]
    
    if sigma == 0:
        # Special case: use the nearest neighbor (minimum distance)
        min_idx = np.argmin(k_distances)
        return k_targets[min_idx]
    else:
        # Compute Gaussian weights: exp(-d^2 / (2*sigma^2))
        weights = np.exp(-(k_distances ** 2) / (2 * sigma ** 2))
        if np.sum(weights) == 0:
            return np.mean(k_targets)
        return np.sum(weights * k_targets) / np.sum(weights)

# Candidate parameter values
k_values = np.arange(1,17,2)
sigma_values = np.arange(0, 2.04, 0.04)  # sigma from 0 to 2 with step 0.04

# Grid search: find best k and sigma based on test MSE
best_mse = float('inf')
best_k = None
best_sigma = None

for k in k_values:
    for sigma in sigma_values:
        # Compute predictions for the test set using current parameters
        y_pred_test = np.array([knn_regression_gaussian(np.array([x]), X_train, y_train, k=k, sigma=sigma)
                                  for x in x_test])
        current_mse = mse(y_test, y_pred_test)
        print(f"k={k}, sigma={sigma:.2f}, Test MSE = {current_mse:.4f}")
        if current_mse < best_mse:
            best_mse = current_mse
            best_k = k
            best_sigma = sigma

print("\nBest combination:")
print("k =", best_k, ", sigma =", best_sigma, ", Testing MSE =", best_mse)

# Visualization with the best parameters
x_range = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 300)
y_range = np.array([knn_regression_gaussian(np.array([x]), X_train, y_train, k=best_k, sigma=best_sigma)
                    for x in x_range])

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Testing Data')
plt.plot(x_range, y_range, color='red', label=f"KNN (k={best_k}, sigma={best_sigma:.2f})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("KNN Regression with Gaussian Kernel Weighting")
plt.legend()
plt.show()
