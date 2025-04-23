from .base import Optimizer
import random
import numpy as np

class HGSA(Optimizer):
    def __init__(self, config):
        super().__init__(config)

    def optimize(self, bot):
        pop = [[random.uniform(self.bounds[d][0], self.bounds[d][1])
                 for d in range(self.dim)] for _ in range(self.pop_size)]
        temps = [1.0]*self.pop_size
        g_best, g_val = pop[0], -float('inf')

        for it in range(self.max_iter):
            scores = [bot.evaluate(ind, bot) for ind in pop]
            # update global best
            for ind, sc in zip(pop, scores):
                if sc > g_val:
                    g_best, g_val = ind[:], sc
            # GA-style evolution
            new_pop = []
            while len(new_pop) < self.pop_size:
                a, b = random.sample(pop, 2)
                cut = random.randint(1, self.dim-1)
                child = a[:cut] + b[cut:]
                # mutation
                for d in range(self.dim):
                    if random.random() < 0.1:
                        perturb = random.uniform(-1, 1)*(self.bounds[d][1]-self.bounds[d][0])*0.1
                        child[d] = max(self.bounds[d][0], min(child[d]+perturb, self.bounds[d][1]))
                new_pop.append(child)
            pop = new_pop
            # SA-style refinement
            for i in range(self.pop_size):
                cand = pop[i][:]
                for d in range(self.dim):
                    cand[d] += np.random.normal(0, temps[i]) * (self.bounds[d][1]-self.bounds[d][0])*0.05
                    cand[d] = max(self.bounds[d][0], min(cand[d], self.bounds[d][1]))
                if bot.evaluate(cand, bot) > bot.evaluate(pop[i], bot):
                    pop[i] = cand
            temps = [t*0.95 for t in temps]       # cool
            print(f"HGSA iter {it+1}/{self.max_iter}  best={g_val:.2f}")
        return g_best
    
    def __str__(self):
        return f"HGSA(pop_size={self.pop_size}, max_iter={self.max_iter})"
