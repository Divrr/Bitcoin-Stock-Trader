import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data
import config
from evaluator import Evaluator

def plot_function_space(train, test):
    sma1_range = np.linspace(5, 500, 500)
    sma2_range = np.linspace(5, 500, 500)
    sma1_grid, sma2_grid = np.meshgrid(sma1_range, sma2_range)

    def evaluate_grid(bot):
        bot.mode = "2d_sma"
        grid = np.zeros_like(sma1_grid)
        for i in range(sma1_grid.shape[0]):
            for j in range(sma1_grid.shape[1]):
                grid[i, j] = bot.evaluate([sma1_grid[i, j], sma2_grid[i, j]])
        return grid

    train_grid = evaluate_grid(train)
    test_grid = evaluate_grid(test)

    vmin = min(train_grid.min(), test_grid.min())
    vmax = max(train_grid.max(), test_grid.max())

    # --- 2D Contour Plot ---
    fig = plt.figure(figsize=(14, 6))
    for idx, (grid, label) in enumerate(zip([train_grid, test_grid], ['Train', 'Test'])):
        ax = fig.add_subplot(1, 2, idx + 1)
        contour = ax.contourf(sma1_grid, sma2_grid, grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"{label} Evaluator - 2D Contour", fontsize=14)
        ax.set_xlabel("High Frequency SMA", fontsize=12)
        ax.set_ylabel("Low Frequency SMA", fontsize=12)

    plt.tight_layout()
    plt.savefig("difference_in_test_train.png")
    plt.show()


if __name__ == "__main__":
    d = config.DATA_CFG
    train = load_data(d["csv_path"], d["train_start"], d["train_end"])
    test  = load_data(d["csv_path"], d["test_start"],  d["test_end"])

    mode = d.get("mode", "blend")

    train_bot = Evaluator(train, mode=mode)
    test_bot  = Evaluator(test,  mode=mode)

    plot_function_space(train_bot, test_bot)