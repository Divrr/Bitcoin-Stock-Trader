import numpy as np
from optimizers import PPSO, HGSA, IGWO, ACO, CCS
from evaluator import Evaluator
from config import DATA_CFG, get_search_space
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_data(path, start=None, end=None):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if start: df = df[df.index >= start]
    if end: df = df[df.index <= end]
    return df["close"].values


def run_experiment(pop_list, iter_list, mode="blend"):
    results = []

    
    train_prices = load_data(DATA_CFG["csv_path"], DATA_CFG["train_start"], DATA_CFG["train_end"])
    test_prices  = load_data(DATA_CFG["csv_path"], DATA_CFG["test_start"],  DATA_CFG["test_end"])

    dim, bounds = get_search_space(mode)

    for pop_size in pop_list:
        for max_iter in iter_list:
            print(f"Running: pop_size={pop_size}, max_iter={max_iter}")
            config = {
                "dim": dim,
                "bounds": bounds,
                "pop_size": pop_size,
                "max_iter": max_iter,
                "max_time": None,
                "max_calls": None,
                "patience": None,
                "min_delta": None
            }

            train_bot = Evaluator(train_prices, mode=mode)
            test_bot  = Evaluator(test_prices,  mode=mode)

            optimizer = ACO(config)
            best_params = optimizer.optimize(train_bot)
            test_profit = test_bot.evaluate(best_params)

            results.append({
                "pop_size": pop_size,
                "max_iter": max_iter,
                "test_profit": test_profit
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    pop_sizes = [30, 50, 100, 200]
    max_iters = [10, 30, 50, 80]
    mode = DATA_CFG["mode"]  

    df = run_experiment(pop_sizes, max_iters, mode)
    # df.to_csv("pop_iter_results.csv", index=False)
    print(df)

    # Plotting the results
    import seaborn as sns
    pivot = df.pivot(index="pop_size", columns="max_iter", values="test_profit")
    pivot = pivot.sort_index(ascending=True)
    sns.heatmap(pivot[::-1], annot=True, fmt=".2f", cmap="viridis",
            yticklabels=pivot.index[::-1])
    plt.title(f"Test Profit (mode={mode})")
    plt.ylabel("Population Size")
    plt.xlabel("Max Iterations")
    plt.tight_layout()
    plt.show()
