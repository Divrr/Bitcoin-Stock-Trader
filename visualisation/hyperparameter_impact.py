import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import random
from optimizers import HGSA, ACO, IGWO, PPSO, CCS
from main import load_data
from evaluator import Evaluator
from config import COMMON_CFG, DATA_CFG

def run_optimizer_with_config(optimizer_class, bot, pop_size, max_iter, max_time):
    config = {
        "dim": COMMON_CFG["dim"],
        "bounds": COMMON_CFG["bounds"],
        "pop_size": pop_size,
        "max_iter": max_iter,
        "max_time": max_time
    }
    optimizer = optimizer_class(config)
    best_params = optimizer.optimize(bot)
    best_profit = bot.evaluate(best_params)
    return best_profit

def plot_and_save(x_values, profits_dict, x_label, title, filename):
    plt.figure(figsize=(8, 6))
    for name, profits in profits_dict.items():
        plt.plot(x_values, profits, marker='o', label=name)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Final Profit (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    # Load training data
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])

    # Constants
    fixed_pop_size = 60
    fixed_max_iter = 30
    fixed_max_time = None

    # Common Evaluator instance
    bot = Evaluator(train_data, mode=DATA_CFG["mode"])

    # Define optimizers to test
    optimizers_to_test = {
        "ACO": ACO,
        "HGSA": HGSA,
        "IGWO": IGWO,
        "PPSO": PPSO,
        "CCS": CCS
    }

    ------ 1. Population Size Test ------
    pop_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    profits_popsize = {name: [] for name in optimizers_to_test}

    for pop_size in pop_sizes:
        print(f"Testing Population Size = {pop_size}")
        for name, optimizer in optimizers_to_test.items():
            profit = run_optimizer_with_config(optimizer, bot, pop_size, fixed_max_iter, None)
            profits_popsize[name].append(profit)

    plot_and_save(
        pop_sizes,
        profits_popsize,
        x_label="Population Size",
        title="Population Size vs Profit",
        filename="population_size_vs_profit.png"
    )

    # ------ 2. Max Iterations Test ------
    max_iters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    profits_maxiter = {name: [] for name in optimizers_to_test}

    for max_iter in max_iters:
        print(f"Testing Max Iterations = {max_iter}")
        for name, optimizer in optimizers_to_test.items():
            profit = run_optimizer_with_config(optimizer, bot, fixed_pop_size, max_iter, None)
            profits_maxiter[name].append(profit)

    plot_and_save(
        max_iters,
        profits_maxiter,
        x_label="Max Iterations",
        title="Max Iterations vs Profit",
        filename="max_iterations_vs_profit.png"
    )

    # ------ 3. Max Time Test ------
    max_times = [5, 10, 20, 30]  # seconds
    profits_maxtime = {name: [] for name in optimizers_to_test}

    for max_time in max_times:
        print(f"Testing Max Time = {max_time} seconds")
        for name, optimizer in optimizers_to_test.items():
            profit = run_optimizer_with_config(optimizer, bot, fixed_pop_size, None, max_time)
            profits_maxtime[name].append(profit)

    plot_and_save(
        max_times,
        profits_maxtime,
        x_label="Max Time (seconds)",
        title="Max Time vs Profit",
        filename="max_time_vs_profit.png"
    )

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
