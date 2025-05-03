import numpy as np
import matplotlib.pyplot as plt

def plot_function_space(train, test):
    sma1_range = np.linspace(5, 50, 100)
    sma2_range = np.linspace(5, 50, 100)
    sma1_grid, sma2_grid = np.meshgrid(sma1_range, sma2_range)

    def evaluate_grid(bot):
        bot.mode = "2d_sma"
        grid = np.zeros_like(sma1_grid)
        for i in range(sma1_grid.shape[0]):
            for j in range(sma1_grid.shape[1]):
                grid[i, j] = bot.evaluate([sma1_grid[i, j], sma2_grid[i, j]])
        return grid

    train_grid = evaluate_grid(train)
    test_grid = evaluate_grid(test)

    vmin = min(train_grid.min(), test_grid.min())
    vmax = max(train_grid.max(), test_grid.max())

    # --- 2D Contour Plot ---
    fig = plt.figure(figsize=(14, 6))
    for idx, (grid, label) in enumerate(zip([train_grid, test_grid], ['Train', 'Test'])):
        ax = fig.add_subplot(1, 2, idx + 1)
        contour = ax.contourf(sma1_grid, sma2_grid, grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"{label} Evaluator - 2D Contour", fontsize=14)
        ax.set_xlabel("High Frequency SMA", fontsize=12)
        ax.set_ylabel("Low Frequency SMA", fontsize=12)

    plt.tight_layout()
    plt.savefig("filename.png")
    plt.show()
