import matplotlib.pyplot as plt
import pandas as pd
import copy
from config import DATA_CFG, COMMON_CFG, get_search_space
from main import run_experiment

# 固定测试集结束时间，改变训练集开始时间
train_starts = [
    "2014-11-28",
    "2015-11-28",
    "2016-11-28",
    "2017-11-28",
]

DATA_CFG["mode"] = "blend"  # 固定为blend模式
mode = DATA_CFG["mode"]
dim, bounds = get_search_space(mode)
COMMON_CFG["dim"] = dim
COMMON_CFG["bounds"] = bounds

results = {}
optimizers_set = set()

for start_date in train_starts:
    print(f"\nRunning with training start date: {start_date}")
    DATA_CFG["train_start"] = start_date

    summary, _ = run_experiment(DATA_CFG, COMMON_CFG)
    df = pd.DataFrame(summary)

    results[start_date] = {}
    for _, row in df.iterrows():
        opt = row["Optimizer"]
        results[start_date][opt] = row["Test$"]
        optimizers_set.add(opt)

# 构建 DataFrame: index = 日期，列 = Optimizer
df_result = pd.DataFrame(results).T.sort_index()

# 绘图
plt.figure(figsize=(10, 6))
for opt in sorted(optimizers_set):
    plt.plot(df_result.index, df_result[opt], marker='o', label=opt)

plt.title("Test Profit vs. Train Dataset Size (Blend Mode)")
plt.xlabel("Train Start Date (Earlier = Larger Dataset)")
plt.ylabel("Test Profit ($)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("trainset_size_vs_profit_blend.png")
plt.show()
