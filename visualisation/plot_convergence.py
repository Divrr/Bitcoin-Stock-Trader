import matplotlib.pyplot as plt

def plot_convergence(optimizers, bot):
    results = {}
    for optimizer in optimizers:
        results[optimizer] = optimizer.convergence_curve

    for name, fitness_list in results.items():
        plt.plot(fitness_list, label=name)

    plt.title("Convergence Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()