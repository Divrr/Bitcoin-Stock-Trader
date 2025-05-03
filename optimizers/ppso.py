from .base import Optimizer
import random
import math
import time
import numpy as np

# Adapted from Ghasemi et al diagram
class PPSO(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.convergence_curve = []

    def optimize(self, bot):
        self.bounds = np.array(self.bounds) 
        v_max = np.full((self.pop_size, self.dim), 0.5 * (self.bounds[:, 1] - self.bounds[:, 0]))
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.pop_size, self.dim))
        theta = np.random.uniform(0, 2 * np.pi, size=self.pop_size)

        p_best = pop.copy()
        p_val = np.array([bot.evaluate(p) for p in pop])

        g_val = np.max(p_val)
        g_best = pop[np.argmax(p_val)].copy()

        start_time = time.time()          # for the max_time check  
        calls0     = bot.eval_count      # so we can count only the new evals  
        best_hist  = []                  # “history” of the best fitness at each iteration  


        for it in self._iter_loop():
            for i in range(self.pop_size):
                # ----- Velocity --------------------------------
                c1 = np.abs(np.cos(theta[i]))**2 * np.sin(theta[i])
                c2 = np.abs(np.sin(theta[i]))**2 * np.cos(theta[i])

                v = c1 * (p_best[i] - pop[i]) + c2 * (g_best - pop[i])
                v = np.clip(v, -v_max[i], v_max[i])

                # ----- Position --------------------------------
                pop[i] += v
                pop[i] = np.clip(pop[i], self.bounds[:, 0], self.bounds[:, 1])

                # ----- Evaluation --------------------------------
                val = bot.evaluate(pop[i])
                if val > p_val[i]:
                    p_best[i] = pop[i].copy()
                    p_val[i] = val

                if val > g_val:
                    g_best = pop[i].copy()
                    g_val = val
                
                # ----- theta and max_v --------------------------------
                theta[i] += np.abs(np.cos(theta[i]) + np.sin(theta[i])) * (2 * np.pi)
                v_max[i] = np.abs(np.cos(theta[i]))**2 * (self.bounds[:, 1] - self.bounds[:, 0])
            
            print(f"PPSO iter {it+1}, best={g_val:.2f}", end="\r")
            self.convergence_curve.append(g_val)

            # *****************************************************************************************
            # *                               record & check early-stop                               *
            # *****************************************************************************************
            best_hist.append(g_val)
            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, best_hist):
                break
        
        print()
        return g_best, g_val