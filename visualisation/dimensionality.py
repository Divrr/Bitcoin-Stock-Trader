import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run_experiment
from config import get_search_space
import pandas as pd

MODES = ["2d_sma", "macd", "blend", "21d_macd"]

def make_config(mode):
    dim, bounds = get_search_space(mode)
    
    data_cfg = {
        "csv_path": "data/BTC-Daily.csv",
        "train_start": "2017-01-01",
        "train_end"  : "2019-12-31",
        "test_start" : "2020-01-01",
        "test_end"   : "2022-03-01",
        "mode": mode,
    }

    common_cfg = {
        "dim": dim,
        "bounds": bounds,
        "pop_size": 30,
        "max_iter": 30,
        "max_time": None,
        "max_calls": None,
        "patience": None,
        "min_delta": None,
    }

    return data_cfg, common_cfg

results = []

for mode in MODES:
    print(f"\n\n{'='*30} TESTING MODE: {mode} {'='*30}\n")
    data_cfg, common_cfg = make_config(mode)
    summary, _ = run_experiment(data_cfg, common_cfg)

    for entry in summary:
        entry["Mode"] = mode
        entry["Dim"] = common_cfg["dim"]
        results.append(entry)

df = pd.DataFrame(results)
print("\n\n" + "="*75)
print("AGGREGATE RESULTS BY DIMENSIONALITY")
print("="*75)
grouped = df.groupby("Dim").agg({
    "Avg Eval (ms)": "mean",
    "Test$": "mean"
}).reset_index()
print(grouped.to_string(index=False))