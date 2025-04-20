import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot import Evaluator  
import numpy as np

class CyclicCoordinateSearch:
    def __init__(self, fitness_function, initial_point, dim=1, max_iter=50, lb=0, ub=1):
        self.fitness_function = fitness_function
        self.dim = dim
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.point = np.array(initial_point)
    
    def line_search(self, dim, current_pos, step_fraction, step_count):
        step_size = (self.ub[dim] - self.lb[dim]) * step_fraction
        search_range = np.linspace(
            max(self.lb[dim], current_pos - step_count * step_size),
            min(self.ub[dim], current_pos + step_count * step_size),
            2 * step_count + 1
        )

        best_fitness = self.fitness_function(self.point)
        best_pos = current_pos

        for pos in search_range:
            self.point[dim] = pos
            fitness = self.fitness_function(self.point)
            if fitness < best_fitness:
                best_fitness = fitness
                best_pos = pos

        self.point[dim] = best_pos
        return best_fitness

    def optimize(self, step_fraction=0.5, step_count=5):
        start = time.time()
        best_fitness = self.fitness_function(self.point)

        for iteration in range(self.max_iter):
            for dim in range(self.dim):
                current_pos = self.point[dim]
                best_fitness = self.line_search(dim, current_pos, step_fraction, step_count)

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {best_fitness:.2f}, Current Point: {self.point}")
        end = time.time()
        print(end-start)
        return self.point, best_fitness


lb = [0, 0, 0, 2, 2, 2, 0.1] * 2
ub = [1, 1, 1, 60, 60, 60, 1.0] * 2

# Fitness function (note: we negate profit because IGWO minimizes)
def fitness_function(params):
    try:
        low_filter = params[:7]
        high_filter = params[7:]
        evaluator_train = Evaluator("data/BTC-Daily.csv", start_date="2017-01-01", end_date="2019-12-31")
        evaluator_train.set_filters(low_filter, high_filter)
        evaluator_train.generate_signals()
        return -evaluator_train.calculate_fitness()
    except Exception as e:
        print("Error:", e)
        return float('inf')  # Penalize invalid solutions

optimizer = CyclicCoordinateSearch(fitness_function, [0,0,0,2,2,2,0.1]*2, dim=14, max_iter=50, lb=lb, ub=ub)
best_params, best_score = optimizer.optimize()

