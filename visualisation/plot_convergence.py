import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from optimizers import IGWO, ACO, HGSA, PPSO
from evaluator import Evaluator
from main import load_data

def main():
    CSV_PATH = "data/BTC-Daily.csv"
    train_data = load_data(CSV_PATH, start="2017-01-01", end="2019-12-31")
    bot = Evaluator(train_data, mode="blend")

    config = {
        "dim": 14,
        "bounds": [(0,1)]*3 + [(5,50)]*3 + [(0.1,0.95)] + [(0,1)]*3 + [(5,50)]*3 + [(0.1,0.95)],
        "pop_size": 30,
        "max_iter": 30
    }

    optimizers = [ACO(config), HGSA(config), IGWO(config), PPSO(config)]
    results = {}

    for optimizer in optimizers:
        optimizer.optimize(bot)
        results[optimizer.__class__.__name__] = optimizer.convergence_curve

    for name, fitness_list in results.items():
        plt.plot(fitness_list, label=name)

    plt.title("Convergence Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
