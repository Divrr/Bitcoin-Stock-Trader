import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
from evaluator import sma_kernel, lma_kernel, ema_kernel, wma
from main import load_data
from config import DATA_CFG

def main():
    # Load data
    data = load_data(DATA_CFG["csv_path"],
                     start=DATA_CFG["train_start"],
                     end=DATA_CFG["train_end"])
    """
    # Load data — Zoom into a smaller period (2019 only)
    data = load_data(DATA_CFG["csv_path"],
                           start="2019-01-01",
                           end="2019-12-31") 
    """
    prices = data.values  # numpy array

    # Calculate Moving Averages
    sma_window = 20
    lma_window = 50
    ema_window = 20
    alpha = 0.2

    sma_values = wma(prices, sma_kernel(sma_window))
    lma_values = wma(prices, lma_kernel(lma_window))
    ema_values = wma(prices, ema_kernel(ema_window, alpha))

    # Truncate dates to match moving averages
    dates = data.index[-len(sma_values):]

    # --- First Plot: 2x2 Grid with Only 3 Graphs ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot Close Price + SMA
    ax = axes[0, 0]
    ax.plot(dates, prices[-len(sma_values):], label="Close Price", color="black")
    ax.plot(dates, sma_values, label=f"SMA-{sma_window}", linestyle="--", color="blue")
    ax.set_title(f"Bitcoin Close Price with SMA-{sma_window}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    # Plot Close Price + LMA
    ax = axes[0, 1]
    ax.plot(dates, prices[-len(sma_values):], label="Close Price", color="black")
    ax.plot(dates, lma_values, label=f"LMA-{lma_window}", linestyle="-.", color="orange")
    ax.set_title(f"Bitcoin Close Price with LMA-{lma_window}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    # Plot Close Price + EMA
    ax = axes[1, 0]
    ax.plot(dates, prices[-len(sma_values):], label="Close Price", color="black")
    ax.plot(dates, ema_values, label=f"EMA-{ema_window} (alpha={alpha})", linestyle=":", color="green")
    ax.set_title(f"Bitcoin Close Price with EMA-{ema_window}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    # Leave 4th subplot empty
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # --- Second Plot: Separate Combined Full Plot ---
    plt.figure(figsize=(14,7))
    plt.plot(dates, prices[-len(sma_values):], label="Close Price", color="black")
    plt.plot(dates, sma_values, label=f"SMA-{sma_window}", linestyle="--", color="blue")
    plt.plot(dates, lma_values, label=f"LMA-{lma_window}", linestyle="-.", color="orange")
    plt.plot(dates, ema_values, label=f"EMA-{ema_window} (alpha={alpha})", linestyle=":", color="green")
    plt.title("Bitcoin Price with All Moving Averages Combined", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()