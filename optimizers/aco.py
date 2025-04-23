from .base import Optimizer
import numpy as np

class ACO(Optimizer):
    def __init__(self, config,  evaporation_rate=0.5):
        super().__init__(config)
        self.evaporation_rate = evaporation_rate
        self.pheromones = np.ones(14)

    def sample_parameters(self):
        # Low filter: Fast
        low_params = []
        for i, (low, high) in enumerate(self.bounds[:7]):
            value = np.random.uniform(low, high)
            if i in [3, 4, 5]:  # Offset index of d1~d3
                value = int(round(value))
            low_params.append(value)

        # High filter: Slow
        high_params = []
        for i, (low, high) in enumerate(self.bounds[7:]):
            if i in [3, 4, 5]:  # Offset index of d1~d3
               low += 10  # Longer periods for high filter
            value = np.random.uniform(low, high)
            if i in [3, 4, 5]:
               value = int(round(value))
            high_params.append(value)

        return low_params + high_params


    def optimize(self, bot, eval_fn):
        best_fitness = -float('inf')
        best_params = None

        for it in range(self.max_iter):
            all_params = []
            all_scores = []

            for ant in range(self.pop_size):
                params = self.sample_parameters()
                fitness = bot.evaluate(params, bot)

                all_params.append(params)
                all_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params

            # Pheromone Update
            self.pheromones *= (1 - self.evaporation_rate)  # Expiration
            best_idx = np.argmax(all_scores)
            best_ant = all_params[best_idx]
            for i in range(14):
                self.pheromones[i] += all_scores[best_idx] / 1000.0  # Rewarding better parameters
            print(f"ACO iter {it+1}/{self.max_iter}, best={best_fitness:.2f}")
        return best_params
