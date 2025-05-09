import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from optimizers import IGWO, ACO, HGSA, PPSO, CCS
from evaluator import Evaluator
from main import load_data
from config import COMMON_CFG, DATA_CFG

def evaluate_optimizer_on_test(optimizer_class, seed, config):
    random.seed(seed)
    np.random.seed(seed)

    test_data = load_data(DATA_CFG["csv_path"],
                          start=DATA_CFG["test_start"],
                          end=DATA_CFG["test_end"])
    bot = Evaluator(test_data, mode=DATA_CFG.get("mode", "blend"))

    optimizer = optimizer_class(config)
    best_params = optimizer.optimize(bot)
    best_fitness = bot.evaluate(best_params)  # Evaluate final profit

    return best_fitness


def main():
    seeds = [0, 1, 9, 10, 11, 13, 14, 25, 42]
    optimizers = {
        "IGWO": IGWO,
        "PPSO": PPSO,
        "HGSA": HGSA,
        "ACO": ACO,
        "CCS": CCS
    }

    results = {name: [] for name in optimizers}

    for seed in seeds:
        for name, opt_class in optimizers.items():
            fitness = evaluate_optimizer_on_test(opt_class, seed, COMMON_CFG)
            results[name].append(fitness)
            print(f"Seed {seed}, Optimizer {name}, Test Profit: ${fitness:.2f}")

    print("\n=== Final Test Profit Summary ===")
    for name, values in results.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"{name}: Avg = ${avg:.2f}, Std = ${std:.2f}")

if __name__ == "__main__":
    main()
