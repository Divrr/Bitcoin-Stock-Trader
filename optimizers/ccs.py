from .base import Optimizer
import numpy as np
import time

class CCS(Optimizer):
    def __init__(self, config):
        super().__init__(config)            
        self.convergence_curve = []

    def optimize(self, bot):
        self.current = [0.5 * (b[0] + b[1]) for b in self.bounds] #start at center

        start_time = time.time()
        calls0 = bot.eval_count
        best_point = self.current[:]
        best_val = bot.evaluate(best_point)
        history = []

        for it in self._iter_loop():
            # each dimension
            for d in range(self.dim):
                val = self._line_search(d, self.current, bot)
                if val > best_val:
                    best_val = val
                    best_point = self.current[:]

            self.convergence_curve.append(best_val)
            print(f"CCS iter {it+1}/{self.max_iter}  best={best_val:.2f}", end="\r")

            history.append(best_val)
            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, history):
                break

        return best_point

    def _line_search(self, dim, point, bot, step_fraction=0.5, step_count=5):
        lb, ub = self.bounds[dim]
        span = (ub - lb) * step_fraction
        grid = [min(max(lb, point[dim] + i * span), ub)
                for i in range(-step_count, step_count + 1)]

        best_val = bot.evaluate(point)
        best_coord = point[dim]
        for pos in grid:
            candidate = point[:]
            candidate[dim] = pos
            val = bot.evaluate(candidate)
            if val > best_val:
                best_val = val
                best_coord = pos

        # update point
        point[dim] = best_coord
        return best_val

    def __str__(self):
        return f"CyclicCSS(max_iter={self.max_iter})"