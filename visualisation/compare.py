import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_data
from evaluator import Evaluator
from config import DATA_CFG, COMMON_CFG
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimizers import IGWO, PPSO, ACO, CCS, HGSA

def run_optimizer(optimizer_class, bot, pop_size, max_iter, max_time):
    config = COMMON_CFG
    config["pop_size"] = pop_size
    config["max_iter"] = max_iter

    optimizer = optimizer_class(config)
    best_params = optimizer.optimize(bot)
    return best_params

def compare_optimizers(optimizer_classes, train_bot, test_bot, pop_size=60, max_iter=50, max_time=None, num_runs=100, results_file="results/optimizer_results.csv"):
    all_results = []
    try:
        for optimizer_class in optimizer_classes:
            optimizer_name = optimizer_class.__name__
            print(f"Running optimizer: {optimizer_name}")
            
            for run in range(num_runs):
                print(f"  Trial {run+1}/{num_runs}                       ")
                try:
                    best_params = run_optimizer(optimizer_class, train_bot, pop_size, max_iter, max_time)
                    test_score = test_bot.evaluate(best_params)
                except Exception as e:
                    print(f"    Error in {optimizer_name} run {run+1}: {e}")
                    test_score = float('-inf')  # Or np.nan

                all_results.append({
                    "optimizer": optimizer_name,
                    "best_params": best_params,
                    "run": run + 1,
                    "score": test_score
                })
    except:
        print("EXCEPTION")
    finally:
        if input("IMPORTANT: Do you want to add this to the csv file? (will change the results) (Y if yes) > ") == "Y":
            df = pd.DataFrame(all_results)

            if os.path.exists(results_file): df.to_csv(results_file, mode='a', header=False, index=False)
            else: df.to_csv(results_file, index=False)
            analyze_results()

from scipy.stats import f_oneway, ttest_ind, shapiro, levene
from itertools import combinations
import pandas as pd
import seaborn as sns
import os
import warnings

def analyze_results(results_file="results/optimizer_results.csv"):
    print("Hello")
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file}")
        return

    df = pd.read_csv(results_file)
    df = df[df["score"].notna() & (df["score"] != float('-inf'))]

    print("\n===== Summary Statistics =====")
    print(df.groupby("optimizer")["score"].describe())

    group_scores = [group["score"].values for _, group in df.groupby("optimizer")]
    print("\n===== One-Way ANOVA =====")
    anova_stat, anova_p = f_oneway(*group_scores)
    print(f"F = {anova_stat:.3f}, p = {anova_p:.4f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if anova_p < 0.05:
            print("=> Statistically significant differences detected.")
            print("\n===== Pairwise t-tests (Uncorrected) =====")
            for opt1, opt2 in combinations(df["optimizer"].unique(), 2):
                scores1 = df[df["optimizer"] == opt1]["score"]
                scores2 = df[df["optimizer"] == opt2]["score"]
                stat, p = ttest_ind(scores1, scores2, equal_var=True)  # set False if Levene fails
                print(f"{opt1} vs {opt2}: t = {stat:.2f}, p = {p:.4f}")
        else:
            print("=> No statistically significant difference among groups.")
        

    print("\n===== Showing Boxplot =====")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="optimizer", y="score", palette="Set3", showmeans=True)
    plt.title("Boxplot of Optimizer Scores")
    plt.ylabel("Test Score (Fitness or Profit)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimizer_boxplot.png")
    plt.show()

if __name__ == "__main__":
    intent = str(input("Would you like to [R]un the comparison test, or [V]iew the results? [R/V]"))

    if intent.upper() == "R":
        train_data = load_data(DATA_CFG["csv_path"], start=DATA_CFG["train_start"], end=DATA_CFG["train_end"])
        test_data = load_data(DATA_CFG["csv_path"], start=DATA_CFG["test_start"], end=DATA_CFG["test_end"])
        train_bot = Evaluator(train_data, mode=DATA_CFG["mode"])
        test_bot = Evaluator(test_data, mode=DATA_CFG["mode"])
        compare_optimizers([IGWO, PPSO, ACO, HGSA, CCS], train_bot, test_bot)
    if intent.upper() == "V":
        analyze_results()