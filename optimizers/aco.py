from .base import Optimizer
import numpy as np
import time

class ACO(Optimizer):
    def __init__(self, config,  evaporation_rate=0.5):
        super().__init__(config)
        self.config = config
        self.evaporation_rate = evaporation_rate
        self.bounds = config["bounds"]
        self.pheromones = np.ones(config["dim"])  # pheromone levels for each parameter
        self.convergence_curve = []


    def sample_parameters(self):
        params = []
        int_indices = []
        dim = self.config["dim"]
        if dim == 14:
            int_indices = [3,4,5,10,11,12]
        elif dim == 21:
            int_indices = [3,4,5,10,11,12,17,18,19]
        elif dim == 2:
            int_indices = [0,1]
        elif dim == 3 or dim == 4:  
            int_indices = [0, 1, 2]
        for i, (low, high) in enumerate(self.bounds):
            value = np.random.uniform(low, high)
            if i in int_indices:
                value = int(round(value))
            params.append(value)
        return params


    def optimize(self, bot):
        best_fitness = -float('inf')
        best_params = None

        start_time = time.time()          # for the max_time check  
        calls0     = bot.eval_count      # so we can count only the new evals  
        best_hist  = []                  # “history” of the best fitness at each iteration  


        for it in self._iter_loop():
            all_params = []
            all_scores = []

            for _ in range(self.pop_size):
                params = self.sample_parameters()
                fitness = bot.evaluate(params)

                all_params.append(params)
                all_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params

            # Pheromone Update
            self.pheromones *= (1 - self.evaporation_rate)  # Expiration
            best_idx = np.argmax(all_scores)
            # best_ant = all_params[best_idx]
            for i in range(self.config["dim"]):
                self.pheromones[i] += all_scores[best_idx] / 1000.0  # Rewarding better parameters
            print(f"ACO iter {it+1}/{self.max_iter}, best={best_fitness:.2f}", end="\r")
            self.convergence_curve.append(best_fitness)

            

            # *****************************************************************************************
            # *                               record & check early-stop                               *
            # *****************************************************************************************
            best_hist.append(best_fitness)
            calls_made = bot.eval_count - calls0
            if self._should_stop(start_time, calls_made, best_hist):
                break
        
        print("")
        return best_params
