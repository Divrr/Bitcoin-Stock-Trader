import pandas as pd
from optimizers import PPSO, HGSA, Optimizer
from evaluator import Evaluator, evaluation_function

def load_data(csv_path, start=None, end=None, price_col="close"):
    df = (pd.read_csv(csv_path, parse_dates=["date"])
            .rename(columns=str.lower)
            .set_index("date")
            .sort_index())
    if start or end:
        df = df.loc[start:end]
    return df[price_col]

def run_experiment(optimizer:Optimizer, trainer, tester, params_config):
    best_params = optimizer.optimize(trainer, evaluation_function, params_config["dim"], params_config["bounds"])
    test_performance = evaluation_function(best_params, tester)
    
    return {
        "name": optimizer,
        "best_params": best_params,
        "test_performance": test_performance
    }

def main():
    CSV_PATH = "data/BTC-Daily.csv"
    
    train_data = load_data(CSV_PATH, start="2017-01-01", end="2019-12-31")
    test_data = load_data(CSV_PATH, start="2019-12-31", end="2022-01-01")
    train_bot = Evaluator(train_data, mode="blend")
    test_bot = Evaluator(test_data, mode="blend")

    params_config = {
        "dim": 14,
        "bounds": [
            # HIGH parameters (first 7)
            (0, 1),
            (0, 1),
            (0, 1),
            (5, 50),
            (5, 50),
            (5, 50),
            (0.1, 0.95),

            # LOW parameters (last 7)
            (0, 1),
            (0, 1),
            (0, 1),
            (5, 50),
            (5, 50),
            (5, 50),
            (0.1, 0.95),
        ]
    }

    optimizers = [PPSO(pop_size=100, max_iter=50), HGSA(pop_size=100, max_iter=50)]
    results = []
    for optimizer_class in optimizers:
        result = run_experiment(optimizer_class, train_bot, test_bot, params_config)
        results.append(result)
    
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"Best parameters found: {[round(p, 2) for p in result['best_params']]}")
        print(f"Test performance: {result['test_performance']:.2f}")

if __name__ == "__main__":
    main()

