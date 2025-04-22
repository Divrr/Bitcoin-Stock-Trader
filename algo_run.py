import os
import time
import numpy as np
from ACO.aco import ACOOptimizer
from IGWO.igwo import IGWO
from bot import Evaluator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def run_igwo():
    lb = [0, 0, 0, 2, 2, 2, 0.1] * 2
    ub = [1, 1, 1, 60, 60, 60, 1.0] * 2

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
            return float('inf')

    start_time = time.time()
    optimizer = IGWO(fitness_function, dim=14, n_agents=15, max_iter=50, lb=lb, ub=ub)
    best_params, best_score = optimizer.optimize()
    elapsed_time = time.time() - start_time

    if best_params is not None:
        evaluator_test = Evaluator("data/BTC-Daily.csv", start_date="2020-01-01", end_date="2022-12-31")
        evaluator_test.set_filters(best_params[:7], best_params[7:])
        evaluator_test.generate_signals()

        plot_path = "results/igwo_signals.png"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        evaluator_test.plot_signals(save_path=plot_path)
        return plot_path, elapsed_time
    else:
        raise ValueError("IGWO optimization failed. No valid parameters found.")

def run_aco():
    data_path = os.path.abspath(os.path.join("data", "BTC-Daily.csv"))
    start_time = time.time()
    optimizer = ACOOptimizer(Evaluator, data_path, n_ants=20, n_iterations=30)
    best_params, _ = optimizer.run()
    elapsed_time = time.time() - start_time

    final_bot = Evaluator(data_path, start_date="2020-01-01", end_date="2022-12-31")
    final_bot.set_filters(best_params[:7], best_params[7:])
    final_bot.generate_signals()

    plot_path = "results/aco_signals.png"
    if os.path.exists(plot_path):
        os.remove(plot_path)

    final_bot.plot_signals(save_path=plot_path)

    return plot_path, elapsed_time
