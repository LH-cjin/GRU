# Q1:

当我执行wind.py文件，看时间间隔统计，有负值

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-18-52-00-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-18-35-25-image.png)

负间隔：`-1 days +00:10:00`、`-2 days +17:40:00`。

负间隔通常说明：原始数据集没有排序、有重复时间戳、或者时间字段解析有问题。如果直接滑窗，会把不连续的时间段拼成“过去72小时”，模型会学到错误模式。

# A1:

```
file_path = r"D:\个人信息\GRU\数据集\wind_dataset.csv"
df = pd.read_csv(file_path, engine='python')
df['Date Time'] = pd.to_datetime(df['Date Time'], errors='coerce')
df = df.dropna(subset=['Date Time']).copy()
# 先不排序，看负间隔
diff_raw = df['Date Time'].diff()
neg_raw = df.loc[diff_raw < pd.Timedelta(0), ['Date Time']].copy()
neg_raw['diff'] = diff_raw[neg_raw.index]
print("排序前负间隔：", len(neg_raw))
print(neg_raw)

# 再排序，看负间隔
df = df.sort_values('Date Time').reset_index(drop=True)
diff_sorted = df['Date Time'].diff()
neg_sorted = df.loc[diff_sorted < pd.Timedelta(0), ['Date Time']].copy()
neg_sorted['diff'] = diff_sorted[neg_sorted.index]
print("\n排序后负间隔：", len(neg_sorted))
print(neg_sorted)
#- 排序前负间隔 89 个 → 只是原始数据没排序；
#排序后负间隔 0 个 → 时间戳本身没问题；
```

在time.py执行这段代码 ，排序后负间隔消失，说明只是顺序问题，

- 排序前负间隔 89 个 → 只是原始数据没排序；

- 排序后负间隔 0 个 → 时间戳本身没问题；

- 剩下要处理的只是那 6 个正间隔异常（20min、30min、16h、16h50m、3天2h20m）。  

# Q2：

执行time.py文件，执行结果只显示到31.12.2016 23:50:00，而原始数据集中最后一行的时间是01.01.2017 00:00:00，倒数第二行时间才是31.12.2016 23:50:00

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-23-13-52-image.png)

# A2：

我修改代码，`df.tail(3)` 显示最后一行明明是 `01.01.2017 00:00:00`，但 `.min()` / `.max()` 却给出 `31.12.2016 23:50:00`。

这不是数据丢了，也不是解析失败——是因为**打印 min/max 那一步，`Date Time` 列还是字符串（`object` 类型）**，pandas 按**字典序**比较字符串：

- `'01.01.2009 00:10:00'` → 开头是 `0`，最小；

- `'31.12.2016 23:50:00'` → 开头是 `3`，最大；

- `'01.01.2017 00:00:00'` → 开头是 `0`，在字符串排序里反而**小于** `31...`。

所以 `.max()` 返回了 `31.12.2016 23:50:00`。数据本身没丢，420451 行都在，最后一行就是 `01.01.2017 00:00:00`。

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-23-26-37-image.png)

我再次修改代码，输出结果正确。![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-23-30-24-image.png)

# Q3:

在time.py文件中，去掉 327 个重复时间戳,   这 327 行是**同一个时间戳出现了两次**（或以上）的行drop_duplicates(TIME_COL, keep='first')` 只保留每个时间戳第一次出现的行，后面的丢掉。**下一步要确认**这 327 个重复是"完全重复行"还是"时间相同但值不同"。

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-23-48-34-image.png)

# A3

我新建test.py文件，用来测试

```
import pandas as pd
df = pd.read_pickle(r"D:\个人信息\GRU\数据集\初步处理.pkl")
# 注意：此时 df 已经去重，无法直接看
# 想要查重复的原始情况，需要重新读原 csv
raw = pd.read_csv(r"D:\个人信息\GRU\数据集\wind_dataset.csv", engine='python')
raw['Date Time'] = pd.to_datetime(raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
raw = raw.sort_values('Date Time')
dup = raw[raw['Date Time'].duplicated(keep=False)]
print(f"重复行总数：{len(dup)}")
print(f"重复时间戳个数：{dup['Date Time'].nunique()}")
# 挑几个看看，两行是完全一样还是值不同
for t in dup['Date Time'].drop_duplicates().head(3):
    print("\n" + "=" * 70)
    print("时间戳：", t)
    print(raw[raw['Date Time'] == t])
```

运行结果如图，

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-06-23-59-15-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-00-04-06-image.png)

测试结果显示 654行这些都是完全重复行，保留327行，那 `Q3代码中的keep='first'` 完全没问题。

# Q4  处理缺口

在time.py中，输出结果如图，去重后的行数420124 - 1 = 420123`条间隔。

**420117 + 2 + 1 + 1 + 1 + 1 = 420123 = 去重后行数 - 1**，完全对得上。

`负间隔数量：0` 说明Q1问题彻底解决，排序已经到位。

前 3 个（20min、30min）是**小缺口**，可以在段内插值补上；

后 3 个（16h、16h50m、3天2h）是**大断点**，必须把数据切成不同段，绝不能跨过去滑窗。

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-00-19-23-image.png)

下图是原始数据集，显然，一个30min，两个20min

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-12-39-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-17-19-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-19-23-image.png)

## 小缺口（20min、30min）怎么处理

| A. 段内插值        | 把缺的 1~2 个点用前后时刻线性插出来 | 段变长，窗口更多，但引入少量合成数据 |
| -------------- | -------------------- | ------------------ |
| **B. 段内滑窗时跳过** | 不插值，只是滑窗时避开跨缺口的位置    | 数据全是真的，但窗口少一点点     |

**采用方案 A**，因为：

- 只缺 3 个点（2 个 20min + 1 个 30min，共缺 4 个 10min 点）；

- 占总数据 420124 行的 **0.001%**，几乎可以忽略；

- 插值方式用"时间线性"最自然，跟气象数据 10min 采样间隔非常接近。
  
  1. 缺口只有 3 处，共 4 个点；
  
  2. 气象数据本身有很强的连续性，10min 内 WIND 变化非常小，插值几乎不引入误差；
  
  3. 补完后段内都是完整 10min 网格，滑窗逻辑最简单，不用考虑边界情况；
  
  4. 后续做误差分析、画图都会更干净。

## A4

执行insert.py文件，输出结果如下图所示。

因为有三个大断点，要把"中间有大断点"的数据切成若干个"内部真正连续 10min"的独立块。所以，分为4个段。

统计插值后各段长度，

输入：432 步（72 小时）  想要得到输出：144 步（24 小时）

合计需要 **576 步**（连续 4 天），所以：

段长 < 576  → 一个窗口都滑不出来 → 这个段白搭
段长 = 576  → 只能滑出 1 个样本
段长 = 1728 → 能滑出 1728 - 576 + 1 = 1153 个样本

结论：每一段都远远大于 576，所以**4 段都能用**。

将四个小缺口在各个段内插值补上，插入值 ≈ 前后两行的加权平均。

**新插入的时间戳一定是 10min 网格上的点**（比如 `10:10`、`10:20`），不会是原始存在的时刻；插入行的 14 个特征值都非空；

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-33-11-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-33-33-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-33-54-image.png)

![](C:\Users\Miss.Chen\AppData\Roaming\marktext\images\2026-10-07-16-34-06-image.png)
