import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# =========================================================
# 1. 读取数据
# =========================================================
file_path = "D:\个人信息\GRU\数据集\wind_dataset.csv"
df = pd.read_csv(file_path)
print("原始数据形状：", df.shape)
# 2. 时间处理
# =========================================================
# 按时间排序
df["Date Time"] = pd.to_datetime(
    df["Date Time"],
    format="%d.%m.%Y %H:%M:%S"
)

# 重置索引
df = df.reset_index(drop=True)
print("\n时间范围：")
print("开始时间：", df["Date Time"].iloc[0])
print("结束时间：", df["Date Time"].iloc[-1])

# 3. 检查时间间隔
time_diff = df["Date Time"].diff()
print("\n时间间隔统计：")
print(time_diff.value_counts().head(10))


# =========================================================
# 4. 风向进行 sin / cos 编码
# =========================================================

df["wd_sin"] = np.sin(
    np.deg2rad(df["wd (deg)"])
)
df["wd_cos"] = np.cos(
    np.deg2rad(df["wd (deg)"])
)
# =========================================================
# 5. 定义输入特征
# =========================================================

features = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "WIND",
    "max. wv (m/s)",
    "wd_sin",
    "wd_cos"
]


# =========================================================
# 6. 定义预测目标
# =========================================================

target = "WIND"


print("\n输入特征数量：", len(features))
print("预测目标：", target)


# =========================================================
# 7. 按时间划分数据集
# =========================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]


print("\n数据集划分：")

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# =========================================================
# 8. MinMax归一化
# =========================================================

feature_scaler = MinMaxScaler()

target_scaler = MinMaxScaler()


# 训练集 fit
train_features = feature_scaler.fit_transform(
    train_df[features]
)

train_target = target_scaler.fit_transform(
    train_df[[target]]
)


# 验证集和测试集只能 transform
val_features = feature_scaler.transform(
    val_df[features]
)

val_target = target_scaler.transform(
    val_df[[target]]
)


test_features = feature_scaler.transform(
    test_df[features]
)

test_target = target_scaler.transform(
    test_df[[target]]
)


print("\n归一化完成")


# =========================================================
# 9. 定义时间窗口
# =========================================================

# 10分钟一个点

# 过去72小时
INPUT_LEN = 72 * 60 // 10

# 未来24小时
OUTPUT_LEN = 24 * 60 // 10


print("\n时间窗口：")
print("过去72小时 =", INPUT_LEN, "个时间步")
print("未来24小时 =", OUTPUT_LEN, "个时间步")


# =========================================================
# 10. 创建滑动窗口
# =========================================================

def create_sequences(
        features,
        target,
        input_len,
        output_len
):

    X = []
    y = []

    for i in range(
        len(features)
        - input_len
        - output_len
        + 1
    ):

        # 过去72小时
        X.append(
            features[
                i:i + input_len
            ]
        )

        # 未来24小时
        y.append(
            target[
                i + input_len:
                i + input_len + output_len
            ]
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# =========================================================
# 11. 创建训练/验证/测试数据
# =========================================================

X_train, y_train = create_sequences(
    train_features,
    train_target,
    INPUT_LEN,
    OUTPUT_LEN
)

X_val, y_val = create_sequences(
    val_features,
    val_target,
    INPUT_LEN,
    OUTPUT_LEN
)

X_test, y_test = create_sequences(
    test_features,
    test_target,
    INPUT_LEN,
    OUTPUT_LEN
)


# =========================================================
# 12. 查看最终数据形状
# =========================================================

print("\n最终数据形状：")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)