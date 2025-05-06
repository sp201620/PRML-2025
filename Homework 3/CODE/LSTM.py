import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 数据加载与预处理
def load_and_preprocess(file_path):
    # 加载原始数据
    df = pd.read_csv(
        file_path,
        parse_dates=['date'],
        index_col='date',
        encoding='utf-8'
    )
    print("原始数据样例：\n", df.head())

    # 分离特征类型
    numeric_cols = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
    categorical_cols = ['wnd_dir']

    # 处理缺失值（以pollution列为例）
    df['pollution'].fillna(df['pollution'].median(), inplace=True)

    # 编码分类特征（风向）
    wind_encoder = OneHotEncoder(drop='first', sparse_output=False)
    wind_encoded = wind_encoder.fit_transform(df[categorical_cols])
    wind_cols = wind_encoder.get_feature_names_out(categorical_cols)

    # 标准化数值特征
    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])

    # 合并处理后的特征
    processed_data = np.hstack([numeric_scaled, wind_encoded])
    feature_cols = numeric_cols + wind_cols.tolist()
    
    return pd.DataFrame(processed_data, columns=feature_cols, index=df.index), scaler

# 创建时间序列数据集
def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # 预测pollution（第一个特征）
    return np.array(X), np.array(y)

# 反标准化函数
def inverse_scale(scaler, y_values, feature_index=0):
    """反标准化特定特征"""
    dummy = np.zeros((len(y_values), len(scaler.feature_names_in_)))
    dummy[:, feature_index] = y_values
    return scaler.inverse_transform(dummy)[:, feature_index]

# 主程序
if __name__ == "__main__":
    # 文件路径配置
    file_path = "LSTM-Multivariate_pollution.csv"
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        exit()

    # 数据预处理
    df_processed, scaler = load_and_preprocess(file_path)
    print("\n处理后的特征维度:", df_processed.shape)
    print("特征列表：", df_processed.columns.tolist())

    # 创建时间序列数据集
    n_steps = 24
    X, y = create_dataset(df_processed.values, n_steps)
    print("\n数据集维度:")
    print("X shape:", X.shape)  # (samples, timesteps, features)
    print("y shape:", y.shape)

    # 划分数据集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建LSTM模型
    model = Sequential([
        LSTM(100, activation='relu', 
            input_shape=(n_steps, X.shape[2]),
            return_sequences=True),
        Dropout(0.3),
        LSTM(50, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5)],
        verbose=1
    )

    # 损失曲线可视化
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 预测与评估
    y_pred = model.predict(X_test)
    
    # 反标准化
    y_test_inv = inverse_scale(scaler, y_test)
    y_pred_inv = inverse_scale(scaler, y_pred.flatten())

    # 预测结果可视化
    plt.figure(figsize=(12,6))
    plt.plot(y_test_inv, label='True Values', alpha=0.7)
    plt.plot(y_pred_inv, label='Predictions', alpha=0.7)
    plt.title("PM2.5 Concentration Prediction Comparison")
    plt.xlabel("Time Steps")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 评估指标
    mse = np.mean((y_test_inv - y_pred_inv)**2)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))
    print("\n=== Model Performance ===")
    print(f"MSE:  {mse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")