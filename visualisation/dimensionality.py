import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_data, evaluate_optimizer
from evaluator import Evaluator
from optimizers import ACO, HGSA, IGWO, PPSO, CCS 
from config import get_search_space, COMMON_CFG, DATA_CFG
import pandas as pd

MODES = ["2d_sma", "macd", "blend", "21d_macd"]

def compare_dimensionality():
    common_cfg = COMMON_CFG
    data_cfg = DATA_CFG

    results = []
    train = load_data(data_cfg["csv_path"], data_cfg["train_start"], data_cfg["train_end"])
    test = load_data(data_cfg["csv_path"], data_cfg["test_start"], data_cfg["test_end"])

    for mode in MODES:
        print(f"\nTESTING MODE: {mode.upper()}")

        data_cfg["mode"] = mode
        dim, bounds = get_search_space(mode)
        common_cfg["dim"], common_cfg["bounds"] = dim, bounds

        train_bot = Evaluator(train, mode=mode)
        test_bot = Evaluator(test, mode=mode)

        optimizers = [ACO(common_cfg), HGSA(common_cfg), IGWO(common_cfg), PPSO(common_cfg), CCS(common_cfg)]

        for opt in optimizers:
            metrics, _ = evaluate_optimizer(opt, train_bot, test_bot)
            metrics["Mode"] = mode
            metrics["Optimizer"] = opt
            metrics["dim"] = dim
            results.append(metrics)

    # Create summary DataFrame
    df = pd.DataFrame(results)
    print(f"\n{'='*10}\nAGGREGATE RESULTS BY DIMENSIONALITY\n{'='*10}")
    
    # Ensure numeric columns
    df["Avg Eval (ms)"] = pd.to_numeric(df["Avg Eval (ms)"], errors='coerce')
    df["Test$"] = pd.to_numeric(df["Test$"], errors='coerce')

    grouped = df.groupby(["dim", "Mode"]).agg({
        "Avg Eval (ms)": "mean",
        "Test$": "mean"
    }).reset_index()

    print(grouped.to_string(index=False))

compare_dimensionality()