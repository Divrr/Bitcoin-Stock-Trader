import pandas as pd
import time  # for runtime measurement
from optimizers import PPSO, HGSA, IGWO, ACO
from evaluator import Evaluator


def load_data(csv_path, start=None, end=None, price_col="close"):
    """
    Load and subset CSV data into a sorted price series.
    """
    df = (
        pd.read_csv(csv_path, parse_dates=["date"])  # ensure datetime index
          .rename(columns=str.lower)
          .set_index("date")
          .sort_index()
    )
    if start or end:
        df = df.loc[start:end]
    return df[price_col]


def main():
    CSV_PATH = "data/BTC-Daily.csv"
    # prepare train/test periods per spec
    train_series = load_data(CSV_PATH, start="2017-01-01", end="2019-12-31")
    test_series  = load_data(CSV_PATH, start="2019-12-31", end="2022-01-01")

    # initialize bots
    train_bot = Evaluator(train_series, mode="blend")
    test_bot  = Evaluator(test_series,  mode="blend")

    # optimization hyperparameters\    
    hyperparams = {"dim":14, "bounds":[
        (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
        (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95)
    ], "pop_size":30, "max_iter":30}

    optimizers = [ACO(hyperparams), HGSA(hyperparams), IGWO(hyperparams), PPSO(hyperparams)]
    summary = []  # collect metrics for each optimizer
    best_parameters = {}

    for opt in optimizers:
        # reset counters before each run
        train_bot.eval_count = 0
        train_bot.eval_time  = 0.0

        # optimize on training data and track runtime
        t0 = time.time()
        best_params = opt.optimize(train_bot)
        wall_time = time.time() - t0

        # compute evaluation metrics
        calls       = train_bot.eval_count
        avg_eval_ms = (train_bot.eval_time / calls)*1e3 if calls>0 else 0

        # final test-set evaluation (single call)
        test_fitness = test_bot.evaluate(best_params)

        # we save the best parameters for each optimizer
        best_parameters[opt.__class__.__name__] = best_params

        summary.append({
            'Optimizer':     opt.__class__.__name__,           # name
            'Best Fitness':  round(test_fitness, 2),            # test-set result
            'Iterations':    opt.max_iter,                      # configured
            'Fitness Calls': calls,                             # training calls
            'Wall Time (s':   round(wall_time, 2),               # seconds
            'Avg Eval (ms)':   round(avg_eval_ms, 2)              # ms per eval
        })


    print("\n" + "=" * 50)
    print("FINAL RESULTS COMPARISON")
    print("=" * 50)
    # display consolidated results table
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # we print the best parameters for each optimizer
    print("\n" + "=" * 50)
    print("BEST PARAMETERS FOR EACH OPTIMIZER")
    print("=" * 50)
    for name, params in best_parameters.items():
        # if it is a numpy array, convert it to a list
        if hasattr(params, 'tolist'):
            py_params = params.tolist()
        else:
            py_params = list(params)
        rounded = [round(p, 2) for p in py_params]
        print(f"{name}: {rounded}\n")