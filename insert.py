import pandas as pd
import numpy as np

TIME_COL = 'Date Time'
BEFORE   = r"D:\个人信息\GRU\数据集\初步处理.pkl"
AFTER    = r"D:\个人信息\GRU\数据集\初步处理_插值.pkl"

# =========================================================
# 1. 读取插值前
# =========================================================
df_before = pd.read_pickle(BEFORE)
print("=" * 70)
print(f"插值前：{df_before.shape}")
print("=" * 70)

# =========================================================
# 2. 段内插值
# =========================================================
def fill_segment(g):
    seg_val = g['seg'].iloc[0]
    g = g.drop(columns=['seg'])
    g = g.set_index(TIME_COL)
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq='10min')
    g = g.reindex(full_idx)
    num_cols = g.select_dtypes(include=[np.number]).columns
    g[num_cols] = g[num_cols].interpolate(method='time', limit_direction='both')
    g = g.reset_index().rename(columns={'index': TIME_COL})
    g['seg'] = seg_val
    return g

parts = []
for seg_val, g in df_before.groupby('seg', sort=True):
    parts.append(fill_segment(g))

df_after = pd.concat(parts, ignore_index=True)
df_after = df_after.sort_values(TIME_COL).reset_index(drop=True)

# 重新标段
THRESHOLD = pd.Timedelta('30min')
df_after['seg'] = (df_after[TIME_COL].diff() > THRESHOLD).cumsum()

print(f"插值后：{df_after.shape}")
print("\n插值后各段长度：")
print(df_after.groupby('seg').size())

print("\n段内非 10min 间隔数量（应为 0）：")
for seg, g in df_after.groupby('seg'):
    d = g[TIME_COL].diff().dropna()
    bad = d[d != pd.Timedelta('10min')]
    print(f"  seg {seg}: {len(bad)}")

# =========================================================
# 3. 找出"插值后多出来"的时间戳（= 被插入的 4 行）
# =========================================================
before_times = set(df_before[TIME_COL])
after_times  = set(df_after[TIME_COL])
new_times    = sorted(after_times - before_times)

print("\n" + "=" * 70)
print(f"新插入的时间戳共 {len(new_times)} 个：")
for t in new_times:
    print("  ", t)
print("=" * 70)

# =========================================================
# 4. 打印每个插入点及前后 3 行
# =========================================================
view_cols = [TIME_COL, 'p (mbar)', 'T (degC)', 'Tpot (K)',
             'rh (%)', 'WIND', 'max. wv (m/s)', 'wd (deg)']

for t in new_times:
    idx = df_after.index[df_after[TIME_COL] == t][0]
    print("\n" + "=" * 80)
    print(f"插入点：{t}   （在 df_after 中 index = {idx}）")
    print("=" * 80)

    lo = max(0, idx - 3)
    hi = min(len(df_after), idx + 4)
    block = df_after.iloc[lo:hi][view_cols].copy()
    block['标记'] = ['★ 插入' if x == t else '' for x in block[TIME_COL]]
    print(block.to_string(index=False))

# =========================================================
# 5. 保存
# =========================================================
df_after.to_pickle(AFTER)
df_after.to_csv(r"D:\个人信息\GRU\数据集\初步处理_插值.csv",
                index=False, encoding='utf-8-sig')
print("\n已保存：初步处理_插值.pkl / 初步处理_插值.csv")