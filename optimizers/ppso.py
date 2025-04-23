from .base import Optimizer
import random
import math

class PPSO(Optimizer):
    def __init__(self, config):
        super().__init__(config)

    def initialize(self):
        pop   = [[random.uniform(self.bounds[d][0], self.bounds[d][1])
                  for d in range(self.dim)] for _ in range(self.pop_size)]
        theta = [random.uniform(0, 2*math.pi) for _ in range(self.pop_size)]
        return pop, theta

    def optimize(self, bot):
        pop, theta = self.initialize()
        g_best, g_val = pop[0], -float('inf')

        for it in range(self.max_iter):
            for i in range(self.pop_size):
                # --- PPSO velocity update (phasor rule) ---
                c1 = abs(math.cos(theta[i]))**2 * math.sin(theta[i])
                c2 = abs(math.sin(theta[i]))**2 * math.cos(theta[i])
                r1, r2 = random.random(), random.random()
                v = [0]*self.dim
                p_best = pop[i]                      # no personal best memory for simplicity
                for d in range(self.dim):
                    v[d] = (c1*r1*(p_best[d] - pop[i][d]) +
                            c2*r2*(g_best[d] - pop[i][d]))
                    pop[i][d] += v[d]
                    pop[i][d] = max(self.bounds[d][0], min(pop[i][d], self.bounds[d][1]))
                # evaluate
                val = bot.evaluate(pop[i])
                if val > g_val:
                    g_best, g_val = pop[i][:], val
                # phase angle drift
                theta[i] = (theta[i] + random.uniform(0, 2*math.pi)) % (2*math.pi)
            print(f"PPSO iter {it+1}/{self.max_iter}  best={g_val:.2f}")
        return g_best
    
    def __str__(self):
        return f"PPSO(pop_size={self.pop_size}, max_iter={self.max_iter})"