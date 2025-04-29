import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from optimizers import IGWO, PPSO
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

def main():
    # Load training data
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])

    # Constants
    fixed_pop_size = 30
    fixed_max_iter = 30
    fixed_dim = 14
    fixed_max_time = None  # normally None unless we test max_time

    # Common Evaluator instance
    bot = Evaluator(train_data, mode="blend")

    # ------ 1. Population Size Test ------
    pop_sizes = [10, 20, 30, 40, 50, 60]
    profits_popsize_igwo = []
    profits_popsize_ppso = []

    for pop_size in pop_sizes:
        print(f"Testing Population Size = {pop_size}")
        profit_igwo = run_optimizer_with_config(IGWO, bot, pop_size, fixed_max_iter, None)
        profit_ppso = run_optimizer_with_config(PPSO, bot, pop_size, fixed_max_iter, None)
        profits_popsize_igwo.append(profit_igwo)
        profits_popsize_ppso.append(profit_ppso)

    # ------ 2. Max Iterations Test ------
    max_iters = [10, 20, 30, 50, 100]
    profits_maxiter_igwo = []
    profits_maxiter_ppso = []

    for max_iter in max_iters:
        print(f"Testing Max Iterations = {max_iter}")
        profit_igwo = run_optimizer_with_config(IGWO, bot, fixed_pop_size, max_iter, None)
        profit_ppso = run_optimizer_with_config(PPSO, bot, fixed_pop_size, max_iter, None)
        profits_maxiter_igwo.append(profit_igwo)
        profits_maxiter_ppso.append(profit_ppso)

    # ------ 3. Max Time Test ------
    max_times = [5, 10, 20, 30]  # seconds
    profits_maxtime_igwo = []
    profits_maxtime_ppso = []

    for max_time in max_times:
        print(f"Testing Max Time = {max_time} seconds")
        profit_igwo = run_optimizer_with_config(IGWO, bot, fixed_pop_size, None, max_time)
        profit_ppso = run_optimizer_with_config(PPSO, bot, fixed_pop_size, None, max_time)
        profits_maxtime_igwo.append(profit_igwo)
        profits_maxtime_ppso.append(profit_ppso)

    # ------ Plotting Results ------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns


    # 1. Population Size plot
    axs[0, 0].plot(pop_sizes, profits_popsize_igwo, marker='o', linestyle='-', label="IGWO")
    axs[0, 0].plot(pop_sizes, profits_popsize_ppso, marker='s', linestyle='--', label="PPSO")
    axs[0, 0].set_title("Population Size vs Profit")
    axs[0, 0].set_xlabel("Population Size")
    axs[0, 0].set_ylabel("Final Profit (USD)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)


    # 2. Max Iterations plot
    axs[0, 1].plot(max_iters, profits_maxiter_igwo, marker='o', linestyle='-', label="IGWO")
    axs[0, 1].plot(max_iters, profits_maxiter_ppso, marker='s', linestyle='--', label="PPSO")
    axs[0, 1].set_title("Max Iterations vs Profit")
    axs[0, 1].set_xlabel("Max Iterations")
    axs[0, 1].set_ylabel("Final Profit (USD)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Max Time plot
    axs[1, 0].plot(max_times, profits_maxtime_igwo, marker='o', linestyle='-', label="IGWO")
    axs[1, 0].plot(max_times, profits_maxtime_ppso, marker='s', linestyle='--', label="PPSO")
    axs[1, 0].set_title("Max Time vs Profit")
    axs[1, 0].set_xlabel("Max Time (seconds)")
    axs[1, 0].set_ylabel("Final Profit (USD)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # The empty bottom-right plot (axs[1,1]) — you can leave it blank or hide
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
