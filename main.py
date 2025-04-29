import pandas as pd, psutil, os, time
from optimizers import PPSO, HGSA, IGWO, ACO
from evaluator  import Evaluator
import config  # <- new central settings file

# -------------------------------------------------- data helper ----------

def load_data(path, start=None, end=None, col="close"):
    df = (pd.read_csv(path, parse_dates=["date"]).rename(columns=str.lower)
            .set_index("date").sort_index())
    if start or end:
        df = df.loc[start:end]
    return df[col]

# -------------------------------------------------- main ---------------

def main():
    # ---- data ---------------------------------------------------------
    d = config.DATA_CFG
    train = load_data(d["csv_path"], d["train_start"], d["train_end"])
    test  = load_data(d["csv_path"], d["test_start"],  d["test_end"])

    mode = d.get("mode", "blend")

    train_bot = Evaluator(train, mode=mode)
    test_bot  = Evaluator(test,  mode=mode)

    # ---- set up optimisers -------------------------------------------
    optimisers = [ACO(config.COMMON_CFG), HGSA(config.COMMON_CFG),
                  IGWO(config.COMMON_CFG), PPSO(config.COMMON_CFG)]

    summary = []
    best_parameters = {}

    for opt in optimisers:

        print("\n" + "=" * 75)
        print(f"RUNNING: {opt.__class__.__name__} OPTIMIZER")
        print("-" * 75 )

        # reset counters
        train_bot.eval_count = train_bot.eval_time = 0

        # mem usage before
        proc = psutil.Process(os.getpid())
        mem0 = proc.memory_info().rss/1e6

        # start time
        t0 = time.time()

        # run the optimizer
        best_params = opt.optimize(train_bot)

        # end time
        wall_time_s = time.time() - t0
        # mem usage after
        mem1 = proc.memory_info().rss/1e6

        # compute evaluation metrics
        calls = train_bot.eval_count
        avg_eval_ms   = (train_bot.eval_time/calls)*1e3 if calls else 0

        # final test-set evaluation (single call)
        test_fitness = test_bot.evaluate(best_params)

        # we save the best parameters for each optimizer
        best_parameters[opt.__class__.__name__] = best_params


        summary.append({
            "Optimizer": opt.__class__.__name__,
            "Test$": round(test_fitness,2),
            "Fitness Calls": calls,
            "Total(s)": round(wall_time_s,2),
            "Avg Eval (ms)": round(avg_eval_ms,2),
            "Mem(MB)": round(max(mem0,mem1),1)
        })

    print("\n" + "=" * 75)
    print("FINAL RESULTS COMPARISON")
    print("=" * 75)
    # display consolidated results table
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # we print the best parameters for each optimizer
    print("\n" + "=" * 75)
    print("BEST PARAMETERS FOR EACH OPTIMIZER")
    print("=" * 75)
    for name, params in best_parameters.items():
        # if it is a numpy array, convert it to a list
        if hasattr(params, 'tolist'):
            py_params = params.tolist()
        else:
            py_params = list(params)
        rounded = [round(p, 2) for p in py_params]
        print(f"{name}: {rounded}\n")

if __name__ == "__main__":
    main()
