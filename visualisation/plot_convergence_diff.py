import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
import matplotlib.pyplot as plt

from optimizers import IGWO, ACO, HGSA, PPSO, CCS
from evaluator import Evaluator
from main import load_data
from config import COMMON_CFG, DATA_CFG

def run_optimizer_with_seed(seed, dataset_type="train"):
    random.seed(seed)
    np.random.seed(seed)

    if dataset_type == "train":
        data = load_data(DATA_CFG["csv_path"],
                         start=DATA_CFG["train_start"],
                         end=DATA_CFG["train_end"])
    else:
        data = load_data(DATA_CFG["csv_path"],
                         start=DATA_CFG["test_start"],
                         end=DATA_CFG["test_end"])
    
    mode = DATA_CFG.get("mode", "blend") 
    bot = Evaluator(data, mode=mode)

    optimizers = [ACO(COMMON_CFG), HGSA(COMMON_CFG), IGWO(COMMON_CFG), PPSO(COMMON_CFG), CCS(COMMON_CFG)]
    results = {}

    for optimizer in optimizers:
        optimizer.optimize(bot)
        results[optimizer.__class__.__name__] = optimizer.convergence_curve

    return results

def plot_convergence_grids(seeds, dataset_type="train"):
    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f"Convergence Curves ({dataset_type.capitalize()} Data)", fontsize=18)

    for i, seed in enumerate(seeds):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        results = run_optimizer_with_seed(seed, dataset_type)

        for name, fitness_list in results.items():
            ax.plot(fitness_list, label=name)
        
        ax.set_title(f"Seed = {seed}", fontsize=12)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness")
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12)

    # Add spacing around subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("plot_convergence_diff")
    plt.show()

def main():
    seeds = [0, 1, 10, 13, 14,20, 25, 42, 50]
    print("Using seeds:", seeds)

    plot_convergence_grids(seeds, dataset_type="train")
    plot_convergence_grids(seeds, dataset_type="test")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
