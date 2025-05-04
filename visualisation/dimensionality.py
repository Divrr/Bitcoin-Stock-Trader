import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_data
from evaluator import Evaluator
from optimizers import ACO, HGSA, IGWO, PPSO, CCS 
from config import get_search_space, COMMON_CFG, DATA_CFG
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time

MODES = ["2d_sma", "macd", "blend", "21d_macd"]

def evaluate_optimizer(optimizer, train_bot, test_bot):
    """Run a single optimizer and return performance metrics and best parameters."""
    print(f"{'-'*10}{optimizer}{'-'*10}")

    train_bot.eval_count = 0
    train_bot.eval_time = 0

    proc = psutil.Process(os.getpid())
    mem_start = proc.memory_info().rss / 1e6
    time_start = time.time()

    best_params = optimizer.optimize(train_bot)

    wall_time = time.time() - time_start
    mem_end = proc.memory_info().rss / 1e6

    eval_calls = train_bot.eval_count
    avg_eval_time_ms = (train_bot.eval_time / eval_calls) * 1e3 if eval_calls else 0
    test_fitness = test_bot.evaluate(best_params)

    metrics = {
        "Optimizer": optimizer,
        "Test$": round(test_fitness, 2),
        "Fitness Calls": eval_calls,
        "Total(s)": round(wall_time, 2),
        "Avg Eval (ms)": round(avg_eval_time_ms, 2),
        "Mem(MB)": round(max(mem_start, mem_end), 1),
    }

    return metrics, best_params

def compare_dimensionality():
    common_cfg = COMMON_CFG
    data_cfg = DATA_CFG

    results = []
    train = load_data(data_cfg["csv_path"], data_cfg["train_start"], data_cfg["train_end"])
    test = load_data(data_cfg["csv_path"], data_cfg["test_start"], data_cfg["test_end"])

    for mode in MODES:
        print(f"\nTESTING MODE: {mode.upper()}")

        data_cfg["mode"] = mode
        dim, bounds = get_search_space(mode)
        common_cfg["dim"], common_cfg["bounds"] = dim, bounds

        train_bot = Evaluator(train, mode=mode)
        test_bot = Evaluator(test, mode=mode)

        optimizers = [ACO(common_cfg), HGSA(common_cfg), IGWO(common_cfg), PPSO(common_cfg), CCS(common_cfg)]

        for opt in optimizers:
            metrics, _ = evaluate_optimizer(opt, train_bot, test_bot)
            metrics["Mode"] = mode
            metrics["Optimizer"] = opt
            metrics["dim"] = dim
            results.append(metrics)

    # Create summary DataFrame
    df = pd.DataFrame(results)
    print(f"\n{'='*10}\nAGGREGATE RESULTS BY DIMENSIONALITY\n{'='*10}")
    
    # Ensure numeric columns
    df["Avg Eval (ms)"] = pd.to_numeric(df["Avg Eval (ms)"], errors='coerce')
    df["Test$"] = pd.to_numeric(df["Test$"], errors='coerce')

    grouped = df.groupby(["dim", "Mode"]).agg({
        "Avg Eval (ms)": "mean",
        "Test$": "mean"
    }).reset_index()

    print(grouped.to_string(index=False))
    return pd.DataFrame(results)  # Return the full results

def visualize_results(df):
    # Ensure numeric types
    df["Avg Eval (ms)"] = pd.to_numeric(df["Avg Eval (ms)"], errors='coerce')
    df["Test$"] = pd.to_numeric(df["Test$"], errors='coerce')

    # Convert optimizer objects to their names (if not already strings)
    df["Optimizer"] = df["Optimizer"].apply(lambda x: x.__class__.__name__ if not isinstance(x, str) else x)

    # Plot 1: Test Profit ($) per Mode per Optimizer
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Mode", y="Test$", hue="Optimizer")
    plt.title("Test Profit per Mode per Optimizer")
    plt.ylabel("Test Profit ($)")
    plt.xlabel("Trading Strategy Mode")
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.show()

    # Plot 2: Evaluation Time per Mode per Optimizer
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Mode", y="Avg Eval (ms)", hue="Optimizer")
    plt.title("Average Evaluation Time per Mode per Optimizer")
    plt.ylabel("Avg Eval Time (ms)")
    plt.xlabel("Trading Strategy Mode")
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    result_df = compare_dimensionality()
    visualize_results(result_df)

