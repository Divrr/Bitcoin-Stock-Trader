#!/usr/bin/env python3
# pop_sweep_21d.py
#
# Compare population sizes (100, 200, 300) for the 21‑D MACD strategy.
# Produces barplot: Test Profit vs Population Size (one bar per optimizer in each pop size group).

import sys, os, copy, time, psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main       import load_data
from evaluator  import Evaluator
from optimizers import ACO, HGSA, IGWO, PPSO, CCS
from config     import get_search_space, COMMON_CFG, DATA_CFG

# ---------------------------------------------------------------------
# EXPERIMENT CONSTANTS
# ---------------------------------------------------------------------
POP_SIZES  = [100]     # sweep values
ITERATIONS = 100                 # fixed generations
MODE       = "21d_macd"          # only this mode is tested
# ---------------------------------------------------------------------


def evaluate_optimizer(opt, train_bot, test_bot):
    """Run a single optimiser; return metrics + best parameters."""
    print(f"{'-'*10}{opt.__class__.__name__}{'-'*10}")

    # reset counters
    train_bot.eval_count = 0
    train_bot.eval_time  = 0

    proc       = psutil.Process(os.getpid())
    mem_start  = proc.memory_info().rss / 1e6
    t0         = time.time()

    best_params = opt.optimize(train_bot)

    wall_time   = time.time() - t0
    mem_end     = proc.memory_info().rss / 1e6
    eval_calls  = train_bot.eval_count
    avg_eval_ms = (train_bot.eval_time / eval_calls) * 1e3 if eval_calls else 0
    test_profit = test_bot.evaluate(best_params)

    return {
        "Test$":         round(test_profit, 2),
        "Fitness Calls": eval_calls,
        "Total(s)":      round(wall_time, 2),
        "Avg Eval (ms)": round(avg_eval_ms, 2),
        "Mem(MB)":       round(max(mem_start, mem_end), 1),
    }, best_params


def compare_population():
    """Run 21‑D MACD for every pop size; return results DataFrame."""
    # force mode
    DATA_CFG["mode"] = MODE
    dim, bounds = get_search_space(MODE)

    # load data once
    train = load_data(DATA_CFG["csv_path"], DATA_CFG["train_start"], DATA_CFG["train_end"])
    test  = load_data(DATA_CFG["csv_path"], DATA_CFG["test_start"],  DATA_CFG["test_end"])

    results = []
    for pop in POP_SIZES:
        cfg = copy.deepcopy(COMMON_CFG)
        cfg.update({
            "dim":       dim,
            "bounds":    bounds,
            "pop_size":  pop,
            "max_iter":  ITERATIONS,
            "max_calls": None,
            "patience":  None,
        })

        print(f"\n=== 21‑D MACD with pop_size={pop}, iterations={ITERATIONS} ===")
        train_bot = Evaluator(train, mode=MODE)
        test_bot  = Evaluator(test,  mode=MODE)

        optimisers = [ACO(cfg), HGSA(cfg), IGWO(cfg), PPSO(cfg), CCS(cfg)]

        for opt in optimisers:
            metrics, _ = evaluate_optimizer(opt, train_bot, test_bot)
            metrics.update({
                "Optimizer": opt.__class__.__name__,
                "pop_size":  pop,
            })
            results.append(metrics)

    return pd.DataFrame(results)


def plot_results(df):
    """Barplot: pop_size vs Test$ grouped by Optimizer."""
    df["Test$"] = pd.to_numeric(df["Test$"], errors="coerce")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="pop_size", y="Test$", hue="Optimizer")
    plt.title(f"21‑D MACD — Test Profit vs Population Size (iters={ITERATIONS})")
    plt.xlabel("Population size")
    plt.ylabel("Test Profit ($)")
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_results = compare_population()
    print("\nSUMMARY\n", df_results[["Optimizer", "pop_size", "Test$"]].to_string(index=False))
    plot_results(df_results)
