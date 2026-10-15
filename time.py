import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 1. 读取数据
# =========================================================
file_path = r"D:\个人信息\GRU\数据集\wind_dataset.csv"
df = pd.read_csv(file_path, engine='python')
print("原始形状：", df.shape)
# ---- 先解析时间 ----
df['Date Time'] = pd.to_datetime(
    df['Date Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce'
)
print("时间解析失败：", df['Date Time'].isna().sum())

# ---- 再打印时间范围（此时已是 datetime）----
print("时间范围：", df['Date Time'].min(), "->", df['Date Time'].max())
print("最后 3 行：")
print(df.tail(3)[['Date Time']])

# 2. 解析时间
#数据里的时间格式 01.01.2009 00:10:00，确实是统一的 日.月.年 时:分:秒；
df['Date Time'] = pd.to_datetime(
    df['Date Time'],
    format='%d.%m.%Y %H:%M:%S',
    errors='coerce'
)
n_fail = df['Date Time'].isna().sum()
print(f"\n时间解析失败：{n_fail}")
if n_fail > 0:
    print(df[df['Date Time'].isna()].head(10))

df = df.dropna(subset=['Date Time']).copy()

# 3. 排序 + 去重
df = df.sort_values('Date Time').reset_index(drop=True)
print("\n排序后：", len(df))

before = len(df)
df = df.drop_duplicates('Date Time', keep='first').reset_index(drop=True)
print(f"去重后：{len(df)}  （去掉 {before - len(df)} 个重复时间戳）")

# =========================================================
# 4. 间隔统计
# =========================================================
diff = df['Date Time'].diff()
print("\n排序去重后的间隔统计：")
print(diff.value_counts())

neg = diff[diff < pd.Timedelta(0)]
print(f"\n负间隔数量：{len(neg)}")
if len(neg) > 0:
    print(df.loc[neg.index][['Date Time']])

# =========================================================
# 5. 标记连续段
THRESHOLD = pd.Timedelta('30min')
df['seg'] = (diff > THRESHOLD).cumsum()

print(f"\n段数量：{df['seg'].nunique()}")
print("\n各段长度：")
print(df.groupby('seg').size().sort_values(ascending=False))

print("\n各段起止时间：")
for seg, g in df.groupby('seg'):
    dur_h = (g['Date Time'].max() - g['Date Time'].min()).total_seconds() / 3600
    print(f"seg {seg}: {g['Date Time'].min()} -> {g['Date Time'].max()}  "
          f"({len(g)} 行, {dur_h:.2f} 小时)")

# =========================================================
# 6. 保存清洗后的数据
# =========================================================
# 保存
import os
OUT_DIR = r"D:\个人信息\GRU\数据集"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "初步处理.csv")
OUT_PKL = os.path.join(OUT_DIR, "初步处理.pkl")

# CSV：方便用 Excel 打开检查，中文不乱码
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"CSV    -> {OUT_CSV}")

# pickle：后续脚本读取快，datetime/seg 都保留
df.to_pickle(OUT_PKL)
print(f"pickle -> {OUT_PKL}")





