import pandas as pd, psutil, os, time
from optimizers import PPSO, HGSA, IGWO, ACO, CCS
from evaluator  import Evaluator
import config

def load_data(path, start=None, end=None, col="close"):
    df = (pd.read_csv(path, parse_dates=["date"]).rename(columns=str.lower)
            .set_index("date").sort_index())
    if start or end:
        df = df.loc[start:end]
    return df[col]


def run_experiment(data_cfg, common_cfg):
    d = data_cfg
    train = load_data(d["csv_path"], d["train_start"], d["train_end"])
    test  = load_data(d["csv_path"], d["test_start"],  d["test_end"])

    mode = d.get("mode", "blend")

    train_bot = Evaluator(train, mode=mode)
    test_bot  = Evaluator(test,  mode=mode)

    # ---- set up optimisers -------------------------------------------
    optimisers = [ACO(common_cfg), HGSA(common_cfg),
                  IGWO(common_cfg), PPSO(common_cfg),
                  CCS(common_cfg)]

    summary = []
    best_parameters = {}

    for opt in optimisers:
        print("\n" + "=" * 75)
        print(f"RUNNING: {opt.__class__.__name__} OPTIMIZER")
        print("-" * 75 )

        train_bot.eval_count = train_bot.eval_time = 0
        proc = psutil.Process(os.getpid())
        mem0 = proc.memory_info().rss/1e6
        t0 = time.time()

        best_params = opt.optimize(train_bot)

        wall_time_s = time.time() - t0
        mem1 = proc.memory_info().rss/1e6

        calls = train_bot.eval_count
        avg_eval_ms = (train_bot.eval_time/calls)*1e3 if calls else 0
        test_fitness = test_bot.evaluate(best_params)

        best_parameters[opt.__class__.__name__] = best_params

        summary.append({
            "Optimizer": opt.__class__.__name__,
            "Test$": round(test_fitness,2),
            "Fitness Calls": calls,
            "Total(s)": round(wall_time_s,2),
            "Avg Eval (ms)": round(avg_eval_ms,2),
            "Mem(MB)": round(max(mem0,mem1),1),
        })

    return summary, best_parameters

def main():
    summary, best_parameters = run_experiment(config.DATA_CFG,  config.COMMON_CFG)

    print("\n" + "=" * 75)
    print("FINAL RESULTS COMPARISON")
    print("=" * 75)
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("BEST PARAMETERS FOR EACH OPTIMIZER")
    print("=" * 75)
    for name, params in best_parameters.items():
        if hasattr(params, 'tolist'):
            py_params = params.tolist()
        else:
            py_params = list(params)
        rounded = [round(p, 2) for p in py_params]
        print(f"{name}: {rounded}\n")


if __name__ == "__main__":
    main()
