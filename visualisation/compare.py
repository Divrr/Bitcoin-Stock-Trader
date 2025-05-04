import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_data
from evaluator import Evaluator
from config import DATA_CFG, COMMON_CFG
import numpy as np
import matplotlib.pyplot as plt

def run_optimizer(optimizer_class, bot, pop_size, max_iter, max_time):
    config = COMMON_CFG
    config["pop_size"] = pop_size
    config["max_iter"] = max_iter

    optimizer = optimizer_class(config)
    best_params = optimizer.optimize(bot)
    return best_params

def compare_optimizers(optimizer_classes, train_bot, test_bot, pop_size=30, max_iter=50, max_time=None, num_runs=60):
    results = {}

    for optimizer_class in optimizer_classes:
        optimizer_name = optimizer_class.__name__
        print(f"Running optimizer: {optimizer_name}")
        test_scores = []

        for run in range(num_runs):
            print(f"  Trial {run+1}/{num_runs}")
            try:
                best_params = run_optimizer(optimizer_class, train_bot, pop_size, max_iter, max_time)
                test_score = test_bot.evaluate(best_params)
                test_scores.append(test_score)
            except Exception as e:
                print(f"    Error in {optimizer_name} run {run+1}: {e}")
                test_scores.append(float('-inf'))  # Or np.nan, depending on your tolerance

        results[optimizer_name] = test_scores

    # ---- Plotting Violin Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [results[name] for name in results]
    labels = list(results.keys())

    ax.boxplot(data, showmeans=True)
    ax.set_title(f"Optimizer Comparison on Test Bot over {num_runs} Runs")
    ax.set_ylabel("Final Profit (or Fitness)")
    ax.grid(True)
    plt.savefig("HELP.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from optimizers import IGWO, PPSO, ACO, CCS, HGSA # Add others as needed

    # Load data and create train/test bots
    train_data = load_data(DATA_CFG["csv_path"], start=DATA_CFG["train_start"], end=DATA_CFG["train_end"])
    test_data = load_data(DATA_CFG["csv_path"], start=DATA_CFG["test_start"], end=DATA_CFG["test_end"])
    train_bot = Evaluator(train_data, mode=DATA_CFG["mode"])
    test_bot = Evaluator(test_data, mode=DATA_CFG["mode"])

    # Compare
    compare_optimizers([IGWO, PPSO, ACO, HGSA, CCS], train_bot, test_bot)