import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
from main import run_experiment
from config import DATA_CFG, COMMON_CFG, get_search_space


modes = {
    "macd": "macd",
    "2d": "2d_sma",
    "14d": "blend",
    "21d": "21d_macd"
}

optimizers = []
memory_records = {}
time_records = {}

for mode_label, mode_key in modes.items():
    print(f"\nRunning mode: {mode_label} ({mode_key})")
    DATA_CFG["mode"] = mode_key
    dim, bounds = get_search_space(mode_key)
    COMMON_CFG["dim"] = dim
    COMMON_CFG["bounds"] = bounds

    summary, _ = run_experiment(DATA_CFG, COMMON_CFG)
    df = pd.DataFrame(summary)

    for _, row in df.iterrows():
        print(f"{row['Optimizer']} | Mode: {mode_label} | Time: {row['Avg Eval (ms)'] / 1000:.4f} s | Mem: {row['Mem(MB)']:.2f} MB")


    if not optimizers:
        optimizers = df["Optimizer"].tolist()

    memory_records[mode_label] = df.set_index("Optimizer")["Mem(MB)"].to_dict()
    time_records[mode_label] = (df.set_index("Optimizer")["Avg Eval (ms)"]).to_dict() 

mem_df = pd.DataFrame(memory_records).T[optimizers]
time_df = pd.DataFrame(time_records).T[optimizers]

# -------------------- Plot -----------------------
plt.figure(figsize=(10, 10))

# Memory plot
plt.subplot(2, 1, 1)
for opt in optimizers:
    plt.plot(mem_df.index, mem_df[opt], marker='o', label=opt)
plt.ylabel("Memory Usage (MB)")
plt.title("Average Memory Usage (MB) per Mode")
plt.legend()

# Time plot
plt.subplot(2, 1, 2)
for opt in optimizers:
    plt.plot(time_df.index, time_df[opt], marker='o', label=opt)
plt.ylabel("Avg Time per Eval (s)")
plt.title("Average Evaluation Time (s) per Mode")
plt.xlabel("Parameter Mode")
plt.legend()

plt.tight_layout()
# plt.savefig("performance_by_mode.png")
plt.show()
