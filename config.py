"""Central place to tweak optimisation/run settings.
Set any field to None to disable that early‑stop rule.
"""

COMMON_CFG = {
    # search space
    "dim": 14,
    "bounds": [
        (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
        (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95)
    ],
    # population + iterations
    "pop_size": 30,
    "max_iter": 100,      # hard cap (can’t be None)

    # ---- optional early‑stopping knobs (None ⇒ ignored) ----
    "max_time" : 120,     # seconds   (e.g. None to ignore)
    "max_calls": None,    # evaluations
    "patience" : 10,      # stagnation window length
    "min_delta": 1.0      # €/USD improvement regarded as progress
}

DATA_CFG = {
    "csv_path": "data/BTC-Daily.csv",
    "train_start": "2017-01-01",
    "train_end"  : "2019-12-31",
    "test_start" : "2019-12-31",
    "test_end"   : "2022-01-01",
}