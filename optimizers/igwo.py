"""
Author: Anushka Dissanayaka Mudiyanselage
Date: April 2025
"""

from .base import Optimizer
import numpy as np
import math
import time

class IGWO(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.lb = np.array([b[0] for b in self.bounds])
        self.ub = np.array([b[1] for b in self.bounds])
        self.convergence_curve = []

    def initialize(self):
        x = np.zeros((self.pop_size, self.dim))
        x[0] = np.random.rand(self.dim)
        for i in range(1, self.pop_size):
            for j in range(self.dim):
                if x[i-1][j] < 0.7:
                    x[i][j] = x[i-1][j] / 0.7
                else:
                    x[i][j] = (1 - x[i-1][j]) / 0.3
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))  # Standard Random Initialization

    def clip_agents(self, agents):
        return np.clip(agents, self.lb, self.ub)

    def gaussian_mutation(self, best_pos):
        mutation = best_pos + np.random.normal(0, 0.1, size=self.dim)
        return self.clip_agents(mutation)

    def optimize(self, bot):
        agents = self.initialize()
        alpha_pos, alpha_score = None, -float("inf")
        beta_pos, beta_score = None, -float("inf")
        delta_pos, delta_score = None, -float("inf")
        
        prev_alpha_score = -float("inf") 
        no_improve = 0

        start_time = time.time()          # for the max_time check  
        calls0     = bot.eval_count      # so we can count only the new evals  
        best_hist  = []                  # “history” of the best fitness at each iteration  



        for iter in self._iter_loop():
            for i in range(self.pop_size):
                fitness = bot.evaluate(agents[i])

                if fitness > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = fitness, agents[i].copy()
                elif fitness > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = fitness, agents[i].copy()
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, agents[i].copy()

            # Cosine adaptive control factor
            # a = 2 * np.cos((iter / self.max_iter) * (np.pi / 2))

            # denominator for the cosine schedule
            # if self.max_iter is not None, then denom = self.max_iter, else denom = iter + 1, i.e, 
            # denom now grows with the iteration counter when the run is “infinite”, the coefficient a still decreases smoothly toward 0, 
            # so the grey-wolf step size shrinks just as in the standard algorithm.
            denom = self.max_iter if self.max_iter is not None else (iter + 1)
            print(f"denom={denom}")
            a = 2 * np.cos((iter/int(denom)) * (np.pi / 2))

            # Position update
            if alpha_pos is not None and beta_pos is not None and delta_pos is not None:
                for i in range(self.pop_size):
                    for j in range(self.dim):
                        r1, r2 = np.random.rand(), np.random.rand()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[j] - agents[i][j])
                        X1 = alpha_pos[j] - A1 * D_alpha

                        r1, r2 = np.random.rand(), np.random.rand()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[j] - agents[i][j])
                        X2 = beta_pos[j] - A2 * D_beta

                        r1, r2 = np.random.rand(), np.random.rand()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[j] - agents[i][j])
                        X3 = delta_pos[j] - A3 * D_delta

                        agents[i][j] = (X1 + X2 + X3) / 3

            agents = self.clip_agents(agents)

            if alpha_pos is not None:
                mutated_alpha = self.gaussian_mutation(alpha_pos)
                mutated_score = bot.evaluate(mutated_alpha)
                if mutated_score > alpha_score:
                    alpha_score = mutated_score
                    alpha_pos = mutated_alpha

                

            print(f"IGWO iter {iter + 1}/{self.max_iter} best={alpha_score:.2f}")
            self.convergence_curve.append(alpha_score)


            # record & check early-stop
            best_hist.append(alpha_score)
            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, best_hist):
                break

        return alpha_pos
