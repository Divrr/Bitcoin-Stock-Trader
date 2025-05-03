import os
import time
import psutil
import pandas as pd

from optimizers import PPSO, HGSA, IGWO, ACO, CCS
from visualisation import plot_convergence, plot_function_space
from evaluator import Evaluator
import config

def load_data(path, start=None, end=None, col="close"):
    """Load time-series data from CSV, optionally slicing by date range."""
    df = (
        pd.read_csv(path, parse_dates=["date"])
        .rename(columns=str.lower)
        .set_index("date")
        .sort_index()
    )
    return df.loc[start:end][col] if start or end else df[col]


def evaluate_optimizer(optimizer, train_bot, test_bot):
    """Run a single optimizer and return performance metrics and best parameters."""
    print(f"{'-'*10}{optimizer}{'-'*10}")

    train_bot.eval_count = 0
    train_bot.eval_time = 0

    proc = psutil.Process(os.getpid())
    mem_start = proc.memory_info().rss / 1e6
    time_start = time.time()

    best_params, train_fitness = optimizer.optimize(train_bot)

    wall_time = time.time() - time_start
    mem_end = proc.memory_info().rss / 1e6

    eval_calls = train_bot.eval_count
    avg_eval_time_ms = (train_bot.eval_time / eval_calls) * 1e3 if eval_calls else 0
    test_fitness = test_bot.evaluate(best_params)

    metrics = {
        "Optimizer": optimizer,
        "Train$": round(train_fitness, 2),
        "Test$": round(test_fitness, 2),
        "Fitness Calls": eval_calls,
        "Total(s)": round(wall_time, 2),
        "Avg Eval (ms)": round(avg_eval_time_ms, 2),
        "Mem(MB)": round(max(mem_start, mem_end), 1),
    }

    return metrics, best_params

def main():
    data_cfg = config.DATA_CFG
    common_cfg = config.COMMON_CFG
    train = load_data(data_cfg["csv_path"], data_cfg["train_start"], data_cfg["train_end"])
    test = load_data(data_cfg["csv_path"], data_cfg["test_start"], data_cfg["test_end"])
    mode = data_cfg.get("mode", "blend")

    train_bot = Evaluator(train, mode=mode)
    test_bot = Evaluator(test, mode=mode)

    optimizers = [ACO(common_cfg), HGSA(common_cfg), IGWO(common_cfg), PPSO(common_cfg), CCS(common_cfg)]

    summary = []
    best_parameters = {}

    for opt in optimizers:
        metrics, params = evaluate_optimizer(opt, train_bot, test_bot)
        summary.append(metrics)
        best_parameters[opt] = params

    print(f"\n{'='*75}\nFINAL RESULTS COMPARISON\n{'='*75}")
    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\n{'='*75}\nBEST PARAMETERS FOR EACH OPTIMIZER\n{'='*75}")
    for name, params in best_parameters.items():
        py_params = params.tolist() if hasattr(params, "tolist") else list(params)
        rounded_params = [round(p, 2) for p in py_params]
        print(f"{name}: {rounded_params}\n")

if __name__ == "__main__":
    main()
