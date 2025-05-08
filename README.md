# Bitcoin Trading Bot Optimizer

This project explores the use of **nature-inspired optimization algorithms** to optimize and evaluate trading strategies for Bitcoin, using real historical price data. The bot simulates trading based on technical indicators  and seeks to maximize profit through algorithmic tuning of strategy parameters.

## Project Structure

```
├── data/
│   └── BTC-Daily.csv              # Historical daily price data for Bitcoin
│
├── optimizers/                    # All optimization algorithms
│   ├── igwo.py                    # Improved Grey Wolf Optimizer
│   ├── aco.py                     # Ant Colony Optimization
│   ├── hgsa.py                    # Hybrid Gravitational Search Algorithm
│   ├── ppso.py                    # Particle-based PSO variant
│   ├── base.py                    # Base Optimizer class
│
├── visualization/                # All visual analysis and experiment scripts
│   ├── compare.py                 # Statistical comparison with ANOVA and boxplot
│   ├── compare21_d.py             # Population size sweep for 21D MACD strategy
│   ├── dimensionality.py          # Evaluates optimizers over different strategy modes
│   ├── generate_test_profit.py    # Seed-based testing on unseen data
│   ├── hyperparameter_impact.py   # Effect of pop size, max_iter, max_time
│   ├── plot_convergence.py        # Single convergence plot per optimizer
│   ├── plot_convergence_diff.py   # Grid of convergence plots across seeds
│   ├── plot_function_space_2.py   # Contour plots of parameter landscape
│   ├── plot_moving_averages.py    # Plot of SMA, LMA, EMA overlays
│   └── plot_prices.py             # Raw price plot (train/test)
│
├── config.py                     # Central config for data and optimizer settings
├── evaluator.py                  # Trading logic simulator for all modes
├── main.py                       # Runs full experiment and prints summary
├── requirements.txt              # Required Python packages
```

## Optimization Strategies

Five nature-inspired optimizers are benchmarked:

- **IGWO** – Improved Grey Wolf Optimizer
- **PPSO** – Particle-based PSO variant
- **ACO** – Ant Colony Optimization
- **HGSA** – Hybrid Gravitational Search Algorithm

## Trading Strategies

The trading strategies are parameterized and evaluated via four modes:

- `blend` – Combined weighted average crossover (14D)
- `macd` – Classic MACD-based signals (3D or 4D)
- `2d_sma` – Simple two-window SMA crossover (2D)
- `21d_macd` – Enhanced MACD with custom filters (21D)


## Results & Visualizations

Generated visual insights include:

- Optimizer performance comparisons (boxplots, barplots)
- Evaluation of hyperparameters (population size, iteration count, time)
- Dimensional impact analysis (2D vs 14D vs 21D)
- Fitness landscape plots (contour maps)
- Signal behavior plots for moving averages and MACD



## Contributors

- Anushka Dissanayaka Mudiyanselage
- Hemkrit Ragoonath
- Silvia Gao
- Runtian Liang
- Adib Rohani

---

Feel free to fork, clone, and build upon this work for your own financial research and learning projects!
