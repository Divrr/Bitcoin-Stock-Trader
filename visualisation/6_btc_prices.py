import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
from main import load_data
from config import DATA_CFG

def main():
    train = load_data(DATA_CFG["csv_path"],
                      start=DATA_CFG["train_start"],
                      end=DATA_CFG["train_end"])
    
    test = load_data(DATA_CFG["csv_path"],
                     start=DATA_CFG["test_start"],
                     end=DATA_CFG["test_end"])
    
    # Determine global min and max for y-axis
    global_min = min(train.min(), test.min())
    global_max = max(train.max(), test.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    axes[0].plot(train.index, train.values, color="blue")
    axes[0].set_title("Training Data", fontsize=14)
    axes[0].set_xlabel("Date", fontsize=12)
    axes[0].set_ylabel("Price (USD)", fontsize=12)
    axes[0].set_ylim(global_min, global_max)
    axes[0].grid(True)
    
    axes[1].plot(test.index, test.values, color="red")
    axes[1].set_title("Testing Data", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("Price (USD)", fontsize=12)
    axes[1].set_ylim(global_min, global_max)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("price function.png")
    plt.show()
    
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(f"Training data price range: ${train.min():.2f} - ${train.max():.2f}")
    print(f"Testing period: {test.index.min()} to {test.index.max()}")
    print(f"Testing data price range: ${test.min():.2f} - ${test.max():.2f}")

if __name__ == "__main__":
    main()
