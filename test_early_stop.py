import matplotlib.pyplot as plt
import pandas as pd
import copy
from config import DATA_CFG, COMMON_CFG, get_search_space
from main import run_experiment
import math

modes = {
    "macd": "macd",
    "2d": "2d_sma",
    "14d": "blend",
    "21d": "21d_macd"
}

results = {}
optimizers_set = set()

# Two settings: On and Off patience
for patience_mode in ["with_patience", "no_patience"]:
    for mode_label, mode_key in modes.items():
        print(f"\nRunning mode: {mode_label}, setting: {patience_mode}")
        DATA_CFG["mode"] = mode_key
        dim, bounds = get_search_space(mode_key)

        # Deep copy config and set patience separately
        cfg = copy.deepcopy(COMMON_CFG)
        cfg["dim"] = dim
        cfg["bounds"] = bounds
        cfg["patience"] = 5 
        if patience_mode == "with_patience":
           cfg["patience"] = 5
           cfg["min_delta"] = 1e-2 
        else:
           cfg["patience"] = None
           cfg["min_delta"] = None

        summary, _ = run_experiment(DATA_CFG, cfg)
        df = pd.DataFrame(summary)

        # Store the test$ scores of each optimizer
        if mode_label not in results:
            results[mode_label] = {}
        for _, row in df.iterrows():
            opt = row["Optimizer"]
            if opt not in results[mode_label]:
                results[mode_label][opt] = {}
            results[mode_label][opt][patience_mode] = row["Test$"]
            optimizers_set.add(opt)


optimizers = sorted(optimizers_set)
mode_names = list(modes.keys())

cols = 3
rows = math.ceil(len(optimizers) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
axes = axes.flatten()  

for idx, opt in enumerate(optimizers):
    ax = axes[idx]
    y_with = []
    y_without = []
    for mode in mode_names:
        y_with.append(results[mode].get(opt, {}).get("with_patience", None))
        y_without.append(results[mode].get(opt, {}).get("no_patience", None))

    ax.plot(mode_names, y_with, marker='o', label="Patience=5")
    ax.plot(mode_names, y_without, marker='s', linestyle='--', label="Patience=None")
    ax.set_title(opt)
    ax.set_ylabel("Profit (Test$)")
    ax.grid(True)

    if idx // cols == rows - 1:
        ax.set_xlabel("Mode")
    
    ax.legend()

for i in range(len(optimizers), len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("Profit Comparison with and without Early Stopping", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.9])
# plt.savefig("subplots_profit_vs_patience.png")
plt.show()

