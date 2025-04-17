import numpy as np

class IGWO:
    def __init__(self, fitness_function, dim, n_agents=10, max_iter=50, lb=0, ub=1):
        self.fitness_function = fitness_function
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)

    def tent_map_init(self):
        x = np.zeros((self.n_agents, self.dim))
        x[0] = np.random.rand(self.dim)
        for i in range(1, self.n_agents):
            for j in range(self.dim):
                if x[i-1][j] < 0.7:
                    x[i][j] = x[i-1][j] / 0.7
                else:
                    x[i][j] = (1 - x[i-1][j]) / 0.3
        return self.lb + x * (self.ub - self.lb)

    def clip_agents(self, agents):
        return np.clip(agents, self.lb, self.ub)

    def gaussian_mutation(self, best_pos):
        mutation = best_pos + np.random.normal(0, 0.1, size=self.dim)
        return self.clip_agents(mutation)

    def optimize(self):
        agents = self.tent_map_init()
        alpha_pos, alpha_score = None, float("inf")
        beta_pos, beta_score = None, float("inf")
        delta_pos, delta_score = None, float("inf")

        for iter in range(self.max_iter):
            for i in range(self.n_agents):
                fitness = self.fitness_function(agents[i])

                if fitness < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = alpha_score, alpha_pos
                    alpha_score, alpha_pos = fitness, agents[i].copy()
                elif fitness < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos
                    beta_score, beta_pos = fitness, agents[i].copy()
                elif fitness < delta_score:
                    delta_score, delta_pos = fitness, agents[i].copy()

            # Cosine adaptive control factor
            a = 2 * np.cos((iter / self.max_iter) * (np.pi / 2))

            # Ensure leaders are initialized before attempting update
            if alpha_pos is not None and beta_pos is not None and delta_pos is not None:
                for i in range(self.n_agents):
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
                mutated_score = self.fitness_function(mutated_alpha)
                if mutated_score < alpha_score:
                    alpha_score = mutated_score
                    alpha_pos = mutated_alpha

            print(f"Iteration {iter + 1}/{self.max_iter}, Best Fitness: {alpha_score:.2f}")

        return alpha_pos, alpha_score
