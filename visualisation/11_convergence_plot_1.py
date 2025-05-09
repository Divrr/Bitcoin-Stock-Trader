import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from optimizers import IGWO, ACO, HGSA, PPSO, CCS
from evaluator import Evaluator
from main import load_data
from config import COMMON_CFG, DATA_CFG
import numpy as np
import random

def main():
    train_data = load_data(DATA_CFG["csv_path"],
                           start=DATA_CFG["train_start"],
                           end=DATA_CFG["train_end"])

    mode = DATA_CFG.get("mode", "blend") 
    bot = Evaluator(train_data, mode=mode)

    optimizers = [ACO(COMMON_CFG), HGSA(COMMON_CFG), IGWO(COMMON_CFG), PPSO(COMMON_CFG), CCS(COMMON_CFG)]
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
    random.seed(42)
    np.random.seed(42)
    main()