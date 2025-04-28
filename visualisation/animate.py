import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optimizers import IGWO
from evaluator import sma_kernel, ema_kernel, wma
from config import DATA_CFG
from main import load_data

# SimpleEvaluator for 2D optimization (SMA and EMA only)
class SimpleEvaluator:
    def __init__(self, prices, alpha=0.2):
        self.prices = prices
        self.alpha = alpha

    def evaluate(self, params):
        sma_window = int(params[0])
        ema_window = int(params[1])

        sma_values = wma(self.prices, sma_kernel(sma_window))
        ema_values = wma(self.prices, ema_kernel(ema_window, self.alpha))

        min_len = min(len(sma_values), len(ema_values))
        sma_values = sma_values[-min_len:]
        ema_values = ema_values[-min_len:]
        prices = self.prices[-min_len:]

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

# Extend IGWO to track agent positions
class TrackingIGWO(IGWO):
    def __init__(self, config):
        super().__init__(config)
        self.positions_history = []

    def optimize(self, bot):
        agents = self.initialize()
        alpha_pos, alpha_score = None, -float("inf")
        beta_pos, beta_score = None, -float("inf")
        delta_pos, delta_score = None, -float("inf")

        for iter in range(self.max_iter):
            for i in range(self.pop_size):
                fitness = bot.evaluate(agents[i])

                if fitness > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = fitness, agents[i].copy()
                elif fitness > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = fitness, agents[i].copy()
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, agents[i].copy()

            a = 2 * np.cos((iter / self.max_iter) * (np.pi / 2))

            if alpha_pos is not None and beta_pos is not None and delta_pos is not None:
                for i in range(self.pop_size):
                    for j in range(self.dim):
                        r1, r2 = np.random.rand(), np.random.rand()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[j] - agents[i][j])
                        X1 = alpha_pos[j] - A1 * D_alpha

                        r1, r2 = np.random.rand(), np.random.rand()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[j] - agents[i][j])
                        X2 = beta_pos[j] - A2 * D_beta

                        r1, r2 = np.random.rand(), np.random.rand()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[j] - agents[i][j])
                        X3 = delta_pos[j] - A3 * D_delta

                        agents[i][j] = (X1 + X2 + X3) / 3

            agents = self.clip_agents(agents)

            # Record positions
            self.positions_history.append(agents.copy())

        return alpha_pos

def main():
    # Load price data
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])
    prices = train_data.values

    bot = SimpleEvaluator(prices)

    config = {
        "dim": 2,
        "bounds": [(5, 50), (5, 50)],  # SMA window, EMA window
        "pop_size": 20,
        "max_iter": 30
    }

    optimizer = TrackingIGWO(config)
    optimizer.optimize(bot)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_xlabel("SMA Window", fontsize=14)
    ax.set_ylabel("EMA Window", fontsize=14)
    ax.set_title("IGWO Agent Movement (SMA vs EMA)", fontsize=16)
    scat = ax.scatter([], [], s=50)

    def update(frame):
        positions = optimizer.positions_history[frame]
        scat.set_offsets(positions[:, :2])
        ax.set_title(f"Iteration {frame+1}")
        return scat,

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(optimizer.positions_history),
                                  interval=500, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
