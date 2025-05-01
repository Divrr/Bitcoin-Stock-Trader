import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from optimizers import IGWO, PPSO
from main import load_data
from evaluator import Evaluator
from config import COMMON_CFG, DATA_CFG

def run_optimizer_with_config(optimizer_class, bot, pop_size, max_iter):
    config = {
        "dim": COMMON_CFG["dim"],
        "bounds": COMMON_CFG["bounds"],
        "pop_size": pop_size,
        "max_iter": max_iter
    }
    optimizer = optimizer_class(config)
    best_params, _ = optimizer.optimize(bot)
    best_profit = bot.evaluate(best_params)
    return best_profit

def main():
    # Load training data
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])
    bot = Evaluator(train_data, mode="blend")

    pop_sizes = [10, 20, 30, 40, 50, 60]

    igwo_profits = []
    ppso_profits = []

    for pop_size in pop_sizes:
        print(f"Testing pop_size={pop_size}")
        profit_igwo = run_optimizer_with_config(IGWO, bot, pop_size, COMMON_CFG["max_iter"])
        profit_ppso = run_optimizer_with_config(PPSO, bot, pop_size, COMMON_CFG["max_iter"])
        igwo_profits.append(profit_igwo)
        ppso_profits.append(profit_ppso)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(pop_sizes, igwo_profits, marker='o', linestyle='-', label='IGWO')
    plt.plot(pop_sizes, ppso_profits, marker='s', linestyle='--', label='PPSO')
    plt.title("Impact of Population Size on Final Profit (IGWO vs PPSO)", fontsize=16)
    plt.xlabel("Population Size", fontsize=14)
    plt.ylabel("Final Profit (USD)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
