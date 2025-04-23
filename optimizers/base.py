from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, config):
        self.pop_size = config["pop_size"]
        self.max_iter = config["max_iter"]
        self.dim = config["dim"]
        self.bounds = config["bounds"]

    @abstractmethod
    def optimize(self, bot, eval_fn, dim, bounds):
        pass

    def __str__(self):
        return f"Optimizer(pop_size={self.pop_size}, max_iter={self.max_iter})"