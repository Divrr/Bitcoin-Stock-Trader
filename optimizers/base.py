from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, pop_size, max_iter):
        self.pop_size = pop_size
        self.max_iter = max_iter

    @abstractmethod
    def optimize(self, bot, eval_fn, dim, bounds):
        pass

    def __str__(self):
        return f"Optimizer(pop_size={self.pop_size}, max_iter={self.max_iter})"