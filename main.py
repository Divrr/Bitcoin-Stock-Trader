import pandas as pd
from optimizers import PPSO, HGSA, IGWO, ACO
from optimizers import Optimizer
from evaluator import Evaluator, evaluation_function

def load_data(csv_path, start=None, end=None, price_col="close"):
    df = (pd.read_csv(csv_path, parse_dates=["date"])
            .rename(columns=str.lower)
            .set_index("date")
            .sort_index())
    if start or end:
        df = df.loc[start:end]
    return df[price_col]

def main():
    CSV_PATH = "data/BTC-Daily.csv"
    
    train_data = load_data(CSV_PATH, start="2017-01-01", end="2019-12-31")
    test_data = load_data(CSV_PATH, start="2019-12-31", end="2022-01-01")
    train_bot = Evaluator(train_data, mode="blend")
    test_bot = Evaluator(test_data, mode="blend")

    hyperparams = {
        "dim": 14,
        "bounds": [
            # HIGH parameters (first 7)
            (0, 1), (0, 1), (0, 1),
            (5, 50), (5, 50), (5, 50),
            (0.1, 0.95),

            # LOW parameters (last 7)
            (0, 1), (0, 1), (0, 1),
            (5, 50), (5, 50), (5, 50),
            (0.1, 0.95),
        ],
        "pop_size": 30,
        "max_iter": 30,
    }

    optimizers = [ACO(hyperparams), HGSA(hyperparams), IGWO(hyperparams), PPSO(hyperparams)]
    results = {}
    for optimizer in optimizers:
        best_params = optimizer.optimize(train_bot, evaluation_function)
        test_performance = evaluation_function(best_params, test_bot)
        results[optimizer] = (best_params,test_performance)

    print("\n" + "=" * 50)
    print("FINAL RESULTS COMPARISON")
    print("=" * 50)
    for optimizer, (best_params, test_performance) in results.items():
        optimizer_name = optimizer.__class__.__name__
        rounded_params = [round(p, 2) for p in best_params]
        print(f"\n{optimizer_name}:")
        print(f"Best parameters found: {rounded_params}")
        print(f"Test performance: {test_performance:.2f}")

if __name__ == "__main__":
    main()

