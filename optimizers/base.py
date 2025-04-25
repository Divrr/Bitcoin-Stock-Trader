from abc import ABC, abstractmethod
import time

class Optimizer(ABC):
    """Common fields + *optional* early‑stopping rules.

    Extra stop‑criteria (all optional, passed via config):
        - max_calls     : hard cap on fitness evaluations
        - max_time      : hard cap on wall‑clock seconds
        - patience      : #iterations with < min_delta improvement allowed
        - min_delta     : smallest fitness improvement counted as progress
    """

    def __init__(self, config):
        # core hyper‑params
        self.pop_size = config["pop_size"]
        self.max_iter = config["max_iter"]
        self.dim      = config["dim"]
        self.bounds   = config["bounds"]

        # optional early‑stop knobs (default: disabled)
        self.max_calls = config.get("max_calls")    # int | None
        self.max_time  = config.get("max_time")     # seconds | None
        self.patience  = config.get("patience")     # int | None
        self.min_delta = config.get("min_delta", 0) # float

    # helper: decide whether to stop, given current state
    def _should_stop(self, start_time, calls_since_start, best_history):
        # time‑based
        if self.max_time is not None and (time.time() - start_time) >= self.max_time:
            return True
        # call‑based
        if self.max_calls is not None and calls_since_start >= self.max_calls:
            return True
        # stagnation‑based
        if self.patience is not None and len(best_history) > self.patience:
            window = best_history[-self.patience:]
            # improvement w.r.t. oldest in window
            if max(window) - window[0] < self.min_delta:
                return True
        return False

    @abstractmethod
    def optimize(self, bot):
        pass
