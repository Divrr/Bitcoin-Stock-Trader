"""
Author: Anushka Dissanayaka Mudiyanselage
Date: April 2025
"""
import sys
import os

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from bot import Evaluator
from igwo import IGWO  
import numpy as np

# Define bounds: 14 dimensions (weights, window sizes, alphas) for low + high filters
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

# Run IGWO
optimizer = IGWO(fitness_function, dim=14, n_agents=15, max_iter=50, lb=lb, ub=ub)
best_params, best_score = optimizer.optimize()

# Convert score back to positive profit
final_profit = -best_score

if best_params is not None:
    print("Low Filter :", best_params[:7])
    print("High Filter:", best_params[7:])
    print(f"Profit on training data: ${-best_score:.2f}")
else:
    print("Optimization failed — no valid solution found.")

# Test performance on unseen data
evaluator_test = Evaluator("data/BTC-Daily.csv", start_date="2020-01-01", end_date="2022-12-31")
evaluator_test.set_filters(best_params[:7], best_params[7:])
evaluator_test.generate_signals()
evaluator_test.plot_signals()
print(f"Profit on test data: ${evaluator_test.calculate_fitness():.2f}")











"""if best_params is not None:
    print("Low Filter :", best_params[:7])
    print("High Filter:", best_params[7:])
    print(f"Profit on training data: ${-best_score:.2f}")

    # Test on unseen data
    evaluator_test = EvaluationFunction("data/BTC-Daily.csv", start_date="2020-01-01", end_date="2022-12-31")
    evaluator_test.set_filters(best_params[:7], best_params[7:])
    evaluator_test.generate_signals()
    
    test_profit = evaluator_test.calculate_fitness()
    print(f"Profit on test data: ${test_profit:.2f}")  # PRINT FIRST
    
    evaluator_test.plot_signals()  # Show plot after print
else:
    print("❌ Optimization failed — no valid solution found.")
"""


