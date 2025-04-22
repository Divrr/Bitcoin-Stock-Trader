import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot import Evaluator
import numpy as np
import random


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "BTC-Daily.csv"))


class ACOOptimizer:
    def __init__(self, evaluator_class, csv_path, n_ants=20, n_iterations=50, evaporation_rate=0.5):
        self.evaluator_class = evaluator_class
        self.csv_path = csv_path
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate

        # Parameter boundaries (weight: 0~1, number of days: 5~50, α: 0.01~0.99)
        self.bounds = [
            (0, 1), (0, 1), (0, 1),     # w1, w2, w3
            (5, 50), (5, 50), (5, 50),  # d1, d2, d3
            (0.01, 0.99)               # a3
        ] * 2  # total: 14 dimensions

        # Initial pheromone concentration
        self.pheromones = np.ones(14)

    def sample_parameters(self):
        # Low filter: Fast
        low_params = []
        for i, (low, high) in enumerate(self.bounds[:7]):
            value = np.random.uniform(low, high)
            if i in [3, 4, 5]:  # Offset index of d1~d3
                value = int(round(value))
            low_params.append(value)

        # High filter: Slow
        high_params = []
        for i, (low, high) in enumerate(self.bounds[7:]):
            if i in [3, 4, 5]:  # Offset index of d1~d3
               low += 10  # Longer periods for high filter
            value = np.random.uniform(low, high)
            if i in [3, 4, 5]:
               value = int(round(value))
            high_params.append(value)

        return low_params + high_params


    def run(self):
        best_fitness = -float('inf')
        best_params = None

        for it in range(self.n_iterations):
            print(f"Iteration {it+1}/{self.n_iterations}", end='\r', flush=True)
            all_params = []
            all_scores = []

            for ant in range(self.n_ants):
                params = self.sample_parameters()
                low_filter = params[:7]
                high_filter = params[7:]

                evaluator = self.evaluator_class(self.csv_path, start_date="2014-11-28", end_date="2019-12-31")
                evaluator.set_filters(low_filter, high_filter)
                evaluator.generate_signals()
                fitness = evaluator.calculate_fitness()

                all_params.append(params)
                all_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params

            # Pheromone Update
            self.pheromones *= (1 - self.evaporation_rate)  # Expiration
            best_idx = np.argmax(all_scores)
            best_ant = all_params[best_idx]
            for i in range(14):
                self.pheromones[i] += all_scores[best_idx] / 1000.0  # Rewarding better parameters

        print("\nOptimization Complete!")
        print("Best Fitness:", best_fitness)
        print("Best Parameters:", best_params)

        return best_params, best_fitness


if __name__ == "__main__":
    optimizer = ACOOptimizer(Evaluator, data_path, n_ants=20, n_iterations=30)
    best_params, best_profit = optimizer.run()

    low_filter = best_params[:7]
    high_filter = best_params[7:]

    # outcome
    final_bot = Evaluator(data_path, start_date="2020-01-01", end_date="2022-03-01")
    final_bot.set_filters(low_filter, high_filter)
    final_bot.generate_signals()
    print("Profit with best params:", final_bot.calculate_fitness()) 
    final_bot.plot_signals()
    final_bot.plot_filters()