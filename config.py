"""Central place to tweak optimisation/run settings.
Set any field to None to disable that early‑stop rule.
"""

DATA_CFG = {
    "csv_path": "data/BTC-Daily.csv",
    "train_start": "2017-01-01",
    "train_end"  : "2019-12-31",
    "test_start" : "2019-12-31",
    "test_end"   : "2022-01-01",
    "mode": "blend",   # # Supports "blend" (14 dimensions), "2d_sma" (2 dimensions), "21d_macd" (21 dimensions), "macd" (3 dimensions)
}

def get_search_space(mode):
    if mode == "blend":
        bounds = [
            (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
            (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95)
        ]
    elif mode == "macd":
        # Standard MACD, three adjustable windows (optional with threshold)
        bounds = [(5, 30), (10, 60), (3, 30)]   # [short, long, signal]
        # If you want 4 dimensions (including threshold), add a histogram threshold range
        # bounds = [(5, 30), (10, 60), (3, 30), (-20, 20)]
    elif mode == "2d_sma":
        bounds = [(5, 50), (10, 100)]
    elif mode == "21d_macd":
        bounds = [
            (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
            (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
            (0,1),(0,1),(0,1),(5,50),(5,50),(5,50),(0.1,0.95),
        ]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return len(bounds), bounds

_dim, _bounds = get_search_space(DATA_CFG.get("mode", "blend"))

COMMON_CFG = {
    # search space
    "dim": _dim,
    "bounds": _bounds,
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


