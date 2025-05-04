from abc import ABC, abstractmethod
import time
import math

class Optimizer(ABC):
    """Common fields + *optional* early-stopping rules.

    Extra stop-criteria (all optional, passed via config):
        - max_calls     : hard cap on fitness evaluations
        - max_time      : hard cap on wall-clock seconds
        - patience      : #iterations with < min_delta improvement allowed
        - min_delta     : smallest fitness improvement counted as progress
    """

    def __init__(self, config):
        # core hyper-params
        self.pop_size = config["pop_size"]
        self.max_iter = config.get("max_iter")      # may be None  ← changed
        self.dim      = config["dim"]
        self.bounds   = config["bounds"]

        # optional early-stop knobs (default: disabled)
        self.max_calls = config.get("max_calls")    # int | None
        self.max_time  = config.get("max_time")     # seconds | None
        self.patience  = config.get("patience")     # int | None
        self.min_delta = config.get("min_delta", 0) # float

    # ------------------------------------------------------------------
    # helper: generator that yields iteration index, respecting max_iter;
    # loops forever if max_iter is None 
    # ------------------------------------------------------------------
    def _iter_loop(self):
        """Yields 0,1,2,…  until self.max_iter (if set)."""
        i = 0
        while self.max_iter is None or i < self.max_iter:
            yield i
            i += 1

    def _max_iter(self):
        """Return numeric cap or math.inf when un-bounded."""
        return self.max_iter if self.max_iter is not None else math.inf


    # helper: decide whether to stop, given current state
    def _should_stop(self, start_time, calls_since_start, best_history):

        # helper: just to print a stopoing criterion message for debugging
        def print_stop():
            print(" ----------------------------------------------------------------------------------- ")
        # time-based
        if self.max_time is not None and (time.time() - start_time) >= self.max_time:
            print_stop()
            print(f"Stopping: max_time {self.max_time} seconds reached\n")
            return True
        # call-based
        if self.max_calls is not None and calls_since_start >= self.max_calls:
            print_stop()
            print(f"Stopping: max_calls {self.max_calls} reached\n")
            return True
        # stagnation-based
        if self.patience is not None and len(best_history) > self.patience:
            window = best_history[-self.patience:]
            if max(window) - window[0] < self.min_delta:
                print_stop()
                print(f"Stopping: no improvement of <{self.min_delta} in last {self.patience} iterations\n")
                return True
        return False

    @abstractmethod
    def optimize(self, bot):
        pass
