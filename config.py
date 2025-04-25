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

    # ---- optional early‑stopping knobs (None ⇒ ignored) ----
    # NOTE: make sure to set at least one of these to a non-None value, else the algorithms will run forever

    "max_iter": 30,          # iterations 
    "max_time" : None,       # seconds   
    "max_calls": None,       # evaluations
    "patience" : None,       # stagnation window length
    "min_delta": None        # € improvement regarded as progress
}

DATA_CFG = {
    "csv_path": "data/BTC-Daily.csv",
    "train_start": "2017-01-01",
    "train_end"  : "2019-12-31",
    "test_start" : "2019-12-31",
    "test_end"   : "2022-01-01",
}