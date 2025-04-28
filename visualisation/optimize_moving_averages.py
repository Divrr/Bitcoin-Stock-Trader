import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from evaluator import sma_kernel, ema_kernel, wma
from optimizers import IGWO
from main import load_data
from config import DATA_CFG

# Find best parameters for moving average strategy using IGWO optimizer
# The parameters being optimized are the SMA window, EMA window, and alpha for the EMA.


def moving_average_fitness(params, prices):
    sma_window = int(params[0])
    ema_window = int(params[1])
    alpha = params[2]

    # Recalculate moving averages
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
    return cash  # Return final profit

class MA_Evaluator:
    def __init__(self, prices):
        self.prices = prices
        self.eval_count = 0

    def evaluate(self, params):
        self.eval_count += 1
        return moving_average_fitness(params, self.prices)

def main():
    # Load training data
    data = load_data(DATA_CFG["csv_path"],
                     start=DATA_CFG["train_start"],
                     end=DATA_CFG["train_end"])

    prices = data.values

    # Create evaluator for optimization
    bot = MA_Evaluator(prices)

    # IGWO optimizer configuration
    config = {
        "dim": 3,  # 3 parameters: SMA window, EMA window, Alpha
        "bounds": [
            (5, 50),   # sma_window
            (5, 50),   # ema_window
            (0.1, 0.95)  # alpha
        ],
        "pop_size": 20,
        "max_iter": 30
    }

    optimizer = IGWO(config)
    best_params = optimizer.optimize(bot)

    print("\n==============================")
    print("Optimization Completed")
    print("==============================")
    print(f"Best Parameters Found:")
    print(f" - SMA window: {int(best_params[0])}")
    print(f" - EMA window: {int(best_params[1])}")
    print(f" - Alpha: {best_params[2]:.3f}")
    print(f"Final Profit: ${bot.evaluate(best_params):.2f}")

if __name__ == "__main__":
    main()


# Comparing optimized SMA, EMA, and alpha parameters helps identify the most effective trading signals 
# for a given market condition. Although financial markets are dynamic and can change over time, optimization 
# provides a strong starting point by maximizing historical profitability. It allows us to adapt strategies more 
# intelligently rather than relying on arbitrary or fixed parameters, and helps in building adaptive trading systems 
# that can adjust to evolving market trends.