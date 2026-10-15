
"""
读取插值后的数据 → 段内滑窗 → 按时间切分 → 只用训练集归一化 → 保存
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
# 0. 配置
# =========================================================
IN_PATH     = r"D:\个人信息\GRU\数据集\初步处理_插值.pkl"
OUT_NPZ     = r"D:\pythonProjectGRU.test1\dataset_scaled.npz"
OUT_XSCALER = r"D:\pythonProjectGRU.test1\x_scaler.pkl"
OUT_YSCALER = r"D:\pythonProjectGRU.test1\y_scaler.pkl"
OUT_META    = r"D:\pythonProjectGRU.test1\meta.pkl"

TIME_COL = 'Date Time'
SEG_COL  = 'seg'
TARGET   = 'WIND'

INPUT_LEN  = 432    # 过去 72 小时 = 432 × 10min
OUTPUT_LEN = 144    # 未来 24 小时 = 144 × 10min
GAP        = 144    # train/val/test 之间的间隔步数（防泄漏）

# 1. 读取
# =========================================================
print("=" * 70)
print("1. 读取数据")
print("=" * 70)
df = pd.read_pickle(IN_PATH)
print(f"形状：{df.shape}")
print(f"时间：{df[TIME_COL].min()} -> {df[TIME_COL].max()}")

feature_cols = [c for c in df.columns if c not in [TIME_COL, SEG_COL]]
assert TARGET in feature_cols, f"{TARGET} 不在特征里！"
target_idx = feature_cols.index(TARGET)

print(f"\n特征数：{len(feature_cols)}")
print(f"特征列表：{feature_cols}")
print(f"{TARGET} 在特征中的位置：{target_idx}")

# =========================================================
# 2. 按段滑窗
# =========================================================
print("\n" + "=" * 70)
print("2. 按段滑窗")
print("=" * 70)

# 2.1 先统计每段能出多少窗口
seg_info = []
for seg_val, g in df.groupby(SEG_COL, sort=True):
    n = len(g)
    n_win = n - INPUT_LEN - OUTPUT_LEN + 1
    if n_win > 0:
        seg_info.append((seg_val, n, n_win))
        print(f"  seg {seg_val}: {n} 行 → {n_win} 个窗口")
    else:
        print(f"  seg {seg_val}: {n} 行 < {INPUT_LEN + OUTPUT_LEN}，跳过")

n_total = sum(x[2] for x in seg_info)
print(f"\n总窗口数：{n_total}")

# 2.2 预分配内存（避免 list append 带来的 2 倍内存开销）
n_features = len(feature_cols)
X = np.empty((n_total, INPUT_LEN, n_features), dtype=np.float32)
y = np.empty((n_total, OUTPUT_LEN, 1),          dtype=np.float32)
t = np.empty(n_total, dtype='datetime64[ns]')
seg_ids = np.empty(n_total, dtype=np.int64)

print(f"\n预分配内存：")
print(f"  X: {X.nbytes / 1024**3:.2f} GB")
print(f"  y: {y.nbytes / 1024**3:.2f} GB")

# 2.3 逐段填充
pos = 0
for seg_val, g in df.groupby(SEG_COL, sort=True):
    g = g.sort_values(TIME_COL).reset_index(drop=True)
    arr_x = g[feature_cols].values.astype(np.float32)
    arr_t = g[TIME_COL].values
    n = len(g)
    n_win = n - INPUT_LEN - OUTPUT_LEN + 1
    if n_win <= 0:
        continue
    for i in range(n_win):
        X[pos] = arr_x[i:i+INPUT_LEN]
        y[pos] = arr_x[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN, target_idx:target_idx+1]
        t[pos] = arr_t[i+INPUT_LEN]       # 预测起点时刻
        seg_ids[pos] = seg_val
        pos += 1

print(f"填充完成：{pos} 个样本")
assert pos == n_total, f"实际 {pos} != 预期 {n_total}"

print(f"\nX: {X.shape}   y: {y.shape}   t: {t.shape}")

# =========================================================
# 3. 按时间切分 train / val / test（留 gap）
# =========================================================
print("\n" + "=" * 70)
print("3. 按时间切分")
print("=" * 70)

# 按预测起点时间排序
order = np.argsort(t)
X, y, t, seg_ids = X[order], y[order], t[order], seg_ids[order]

n = len(X)
t1 = t[int(n * 0.7)]   # 训练集结束时间
t2 = t[int(n * 0.85)]  # 验证集结束时间

gap_td = np.timedelta64(GAP * 10, 'm')  # 144 步 × 10 分钟 = 24 小时

train_mask = t < t1
val_mask   = (t >= t1 + gap_td) & (t < t2)
test_mask  = t >= t2 + gap_td

X_train, y_train, t_train = X[train_mask], y[train_mask], t[train_mask]
X_val,   y_val,   t_val   = X[val_mask],   y[val_mask],   t[val_mask]
X_test,  y_test,  t_test  = X[test_mask],  y[test_mask],  t[test_mask]

print(f"Train: X={X_train.shape}  y={y_train.shape}")
print(f"       时间 {t_train[0]} -> {t_train[-1]}")
print(f"Val  : X={X_val.shape}  y={y_val.shape}")
print(f"       时间 {t_val[0]} -> {t_val[-1]}")
print(f"Test : X={X_test.shape}  y={y_test.shape}")
print(f"       时间 {t_test[0]} -> {t_test[-1]}")

print(f"\n切分点：")
print(f"  t1 = {t1}")
print(f"  t2 = {t2}")
print(f"  gap = {GAP} 步 = {GAP * 10} 分钟")
print(f"  训练集最后: {t_train[-1]}")
print(f"  验证集第一: {t_val[0]}   （应 ≥ {t1 + gap_td}）")
print(f"  验证集最后: {t_val[-1]}")
print(f"  测试集第一: {t_test[0]}   （应 ≥ {t2 + gap_td}）")

# =========================================================
# 4. 只用训练集 fit 归一化器
# =========================================================
# =========================================================
# 4. 归一化（分块版：只用训练集 fit）
# =========================================================
print("\n" + "=" * 70)
print("4. 归一化（只用训练集 fit，分块计算）")
print("=" * 70)

N, T, F = X_train.shape
print(f"训练集：N={N}, T={T}, F={F}")

# ---- 4.1 逐特征计算 mean / std（避免 float64 大数组） ----
print("\n计算每个特征的均值/标准差...")

mean_f = np.zeros(F, dtype=np.float64)
std_f  = np.zeros(F, dtype=np.float64)

for f in range(F):
    # 只复制一个特征，约 N*T*4 bytes ≈ 500 MB，可接受
    vals = X_train[:, :, f].ravel().astype(np.float64)
    mean_f[f] = vals.mean()
    std_f[f]  = vals.std()
    del vals

# 防止 std 为 0 导致除零
std_f[std_f == 0] = 1.0

print(f"每个特征 mean: {mean_f}")
print(f"每个特征 std : {std_f}")

mean32 = mean_f.astype(np.float32)
std32  = std_f.astype(np.float32)

# ---- 4.2 分块应用归一化 ----
def standardize_X(X, mean32, std32, chunk=2000):
    """分块对 (N, T, F) 数组归一化，避免一次性 float64 大数组"""
    out = np.empty_like(X, dtype=np.float32)
    n = X.shape[0]
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        out[i:end] = (X[i:end] - mean32) / std32
    return out

print("\n应用归一化到 X_train / X_val / X_test ...")
X_train_s = standardize_X(X_train, mean32, std32)
X_val_s   = standardize_X(X_val,   mean32, std32)
X_test_s  = standardize_X(X_test,  mean32, std32)

# ---- 4.3 y 归一化（y 小，直接处理） ----
y_mean = float(y_train.mean())
y_std  = float(y_train.std())
if y_std == 0:
    y_std = 1.0
print(f"\ny: mean={y_mean:.4f}  std={y_std:.4f}")

y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
y_val_s   = ((y_val   - y_mean) / y_std).astype(np.float32)
y_test_s  = ((y_test  - y_mean) / y_std).astype(np.float32)

print(f"\n归一化后：")
print(f"  X_train: {X_train_s.shape}  y_train: {y_train_s.shape}")
print(f"  X_val  : {X_val_s.shape}    y_val  : {y_val_s.shape}")
print(f"  X_test : {X_test_s.shape}   y_test : {y_test_s.shape}")

print(f"\n检查（用前 2000 个样本近似）：")
sx = X_train_s[:2000]
sy = y_train_s[:2000]
print(f"  X_train_s 均值 {sx.mean():.4f}  标准差 {sx.std():.4f}")
print(f"  y_train_s 均值 {sy.mean():.4f}  标准差 {sy.std():.4f}")
del sx, sy
# 5. 保存
# =========================================================
print("\n" + "=" * 70)
print("5. 保存")
print("=" * 70)

np.savez(OUT_NPZ,
         X_train=X_train_s, y_train=y_train_s, t_train=t_train,
         X_val=X_val_s,     y_val=y_val_s,     t_val=t_val,
         X_test=X_test_s,   y_test=y_test_s,   t_test=t_test,
         # 保留未归一化目标，评估时可反归一化对照
         y_train_raw=y_train, y_val_raw=y_val, y_test_raw=y_test)

# 保存归一化统计量（用于反归一化）
joblib.dump({'mean': mean_f, 'std': std_f},
            r"D:\pythonProjectGRU.test1\x_scaler_stats.pkl")
joblib.dump({'mean': y_mean, 'std': y_std},
            r"D:\pythonProjectGRU.test1\y_scaler_stats.pkl")

print(f"  x_scaler_stats.pkl")
print(f"  y_scaler_stats.pkl")
joblib.dump({
    'feature_cols': feature_cols,
    'target':       TARGET,
    'target_idx':   target_idx,
    'input_len':    INPUT_LEN,
    'output_len':   OUTPUT_LEN,
    'gap':          GAP,
}, OUT_META)

print(f"已保存：")
print(f"  {OUT_NPZ}")
print(f"  {OUT_XSCALER}")
print(f"  {OUT_YSCALER}")
print(f"  {OUT_META}")

print("\n完成。")