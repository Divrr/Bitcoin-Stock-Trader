# Bitcoin Trading Bot Optimizer

This project explores the use of **nature-inspired optimization algorithms** to optimize and evaluate trading strategies for Bitcoin, using real historical price data. The bot simulates trading based on technical indicators  and seeks to maximize profit through algorithmic tuning of strategy parameters.

## Project Structure

```
├── data/
│   └── BTC-Daily.csv                   # Historical daily price data for Bitcoin
│
├── optimizers/                         # All optimization algorithms
│   ├── igwo.py                         # Improved Grey Wolf Optimizer
│   ├── aco.py                          # Ant Colony Optimization
│   ├── hgsa.py                         # Hybrid Gravitational Search Algorithm
│   ├── ppso.py                         # Particle-based PSO variant
│   ├── ccs.py                          # Cyclic Coordinate Search
│   ├── base.py                         # Base Optimizer class
│
├── visualization/                      # All visual analysis and experiment scripts used in the report
│   ├── 1_dimensionality                # Evaluates optimizers' performance with 2D, 3D, 14D and 21D strategies
│   ├── 2_macd_21d_pop_size.py          # Assesses impact on performance when varying pop_size for 21D MACD strategy
│   ├── 3_assess_dataset_size.py        # Assesses impact on performance when varying starting date of dataset
│   ├── 4_5_plot_2d_function_space.py   # Plots contour of 2D SMA parameter landscape
│   ├── 6_btc_prices.py                 # Displays the bitcoin prices in test and training data
│   ├── 7_test_time_mem.py              # Tests time and memory consumption with 2D, 3D, 14D and 21D strategies
│   ├── 8_9_hyperparameter_impact.py    # Assesses effect of pop_size, max_iter, max_time on performance
│   ├── 10_early_stopping_impact.py     # Assesses effect early stopping mechanisms on performance
│   ├── 11_convergence_plot_1.py        # Single convergence plot per optimizer
│   ├── 11_convergence_plot_9.py        # Grid of convergence plots across seeds
│   ├── 11_convergence_test_profit.py   # Seed-based testing on unseen data
│   └── 12_boxplot_comparison.py        # Statistical comparison of the performance of all algorithms
│
├── config.py                           # Central config for data and optimizer settings
├── evaluator.py                        # Trading logic simulator for all modes
├── main.py                             # Runs full experiment and prints summary
└── requirements.txt                    # Required Python packages
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

## Contributors

- Anushka Dissanayaka Mudiyanselage
- Hemkrit Ragoonath
- Silvia Gao
- Runtian Liang
- Adib Rohani

---

Feel free to fork, clone, and build upon this work for your own financial research and learning projects!

## Acknowledgements
### PPSO Implementation
The implementation of PPSO is heavily inspired by the algorithm flowchart presented in M.Ghasemi, E.Akbari, A.Rahimnejad, S.E.Razavi, S.Ghavidel, andL.Li, “Phasor particle swarm optimization: a simple and efficient variant of pso,” *Soft Computing*, vol. 23, pp. 9701–9718, 2019.

### Use of Artificial Intelligence
While the core implementation and analysis in this project were authored by our team, we used ChatGPT (by OpenAI) selectively as a support tool during development. Specifically, ChatGPT was used for:
- Debugging and troubleshooting when we encountered issues
- Clarifying programming concepts and algorithm design

Contributions generated with ChatGPT were reviewed and validated by the development team to ensure academic integrity and compliance with the University of Western Australia's guidelines for collaborative tools and non-plagiaristic AI usage.
