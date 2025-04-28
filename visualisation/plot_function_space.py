import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from evaluator import sma_kernel, ema_kernel, wma
from main import load_data
from config import DATA_CFG

def moving_average_fitness(sma_window, ema_window, alpha, prices):
    """Fitness function: profit from SMA/EMA crossover strategy."""
    sma_window = int(round(sma_window))
    ema_window = int(round(ema_window))

    sma_values = wma(prices, sma_kernel(sma_window))
    ema_values = wma(prices, ema_kernel(ema_window, alpha))

    min_len = min(len(sma_values), len(ema_values))
    sma_values = sma_values[-min_len:]
    ema_values = ema_values[-min_len:]
    prices = prices[-min_len:]

    signal = np.where(ema_values > sma_values, 1, -1)

    cash, btc = 1000.0, 0.0
    for i in range(len(signal)):
        if signal[i] == 1 and cash > 0:
            btc = cash / prices[i]
            cash = 0
        elif signal[i] == -1 and btc > 0:
            cash = btc * prices[i]
            btc = 0
    if btc > 0:
        cash = btc * prices[-1]
    return cash

def main():
    # Load data
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])

    prices = train_data.values

    # Set fixed alpha
    alpha = 0.2

    # Set parameter sweep range
    sma_range = np.linspace(5, 50, 20)   # 20 values from 5 to 50
    ema_range = np.linspace(5, 50, 20)

    sma_grid, ema_grid = np.meshgrid(sma_range, ema_range)
    fitness_grid = np.zeros_like(sma_grid)

    # Evaluate fitness for each (sma, ema) pair
    for i in range(sma_grid.shape[0]):
        for j in range(sma_grid.shape[1]):
            fitness_grid[i, j] = moving_average_fitness(sma_grid[i, j], ema_grid[i, j], alpha, prices)

    # 3D Surface Plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sma_grid, ema_grid, fitness_grid, cmap='viridis')
    ax.set_title("Fitness Landscape: SMA Window vs EMA Window", fontsize=16)
    ax.set_xlabel("SMA Window", fontsize=14)
    ax.set_ylabel("EMA Window", fontsize=14)
    ax.set_zlabel("Final Profit", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 2D Contour Plot
    plt.figure(figsize=(10,8))
    contour = plt.contourf(sma_grid, ema_grid, fitness_grid, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.title("Fitness Contour: SMA Window vs EMA Window", fontsize=16)
    plt.xlabel("SMA Window", fontsize=14)
    plt.ylabel("EMA Window", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
