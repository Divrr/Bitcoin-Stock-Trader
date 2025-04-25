# Bitcoin-Stock-Trader
CITS4404 project

# bot.py – Trading Bot Evaluation
This module implements the Evaluator class, which simulates and evaluates Bitcoin trading bots based on technical analysis using moving average crossovers.

##      Applied Concepts:
        Technical Analysis (TA): Uses price trends and crossover signals from weighted moving averages to simulate trades.
        Filter Composition: Constructs low and high filters using a blend of SMA, LMA, and EMA components.
        Buy/Sell Signal Generation: Detects crossover points where trading actions are triggered.
        Fitness Evaluation: Calculates profit over historical data, factoring in transaction costs and strategy behavior.

##      Key Functions:
        set_filters(low_filter, high_filter) – Accepts 14 parameters (7 for each filter) to construct trading logic. ([w1, w2, w3, d1, d2, d3, α3,  w4, w5, w6, d4, d5, d6, α4]
)
        generate_signals() – Applies filters to price data to identify buy and sell points.
        calculate_fitness() – Simulates trading over a defined historical period and returns the ending portfolio value.
        plot_filters() and plot_signals() – Visualize filter shapes and signal performance.
        This file serves as the evaluation engine for the optimization algorithm and is fully compatible with the algorithems implementation.


# igwo.py – Improved Grey Wolf Optimizer (IGWO)
This file implements an enhanced version of the Grey Wolf Optimizer (GWO), a nature-inspired, population-based optimization algorithm modeled after the leadership hierarchy and hunting behavior of grey wolves.

##      Applied Concepts::
        Grey Wolf Hierarchy: Simulates Alpha, Beta, and Delta wolves to guide the search.
        Position Update: The remaining wolves update their positions based on the influence of these three leaders.
        Cosine Decreasing Control Parameter (a): Replaces the linear decay in standard GWO to improve balance between exploration and exploitation.
        Tent Map Initialization: A chaotic initialization technique used to enhance the diversity of the initial population, avoiding local optima early on.
        Gaussian Mutation: Applied to the Alpha wolf to further escape stagnation by introducing small random perturbations.

##      Key Functions:
        tent_map_init() – Initializes agents using chaotic tent map logic.
        optimize() – Main optimization loop, handling position updates and convergence.
        gaussian_mutation() – Perturbs the current best solution to explore local neighborhoods.

# igwo_integration.py – IGWO Integration with Trading Bot
This script integrates the IGWO class with the EvaluationFunction trading bot. It is responsible for defining the fitness function and running the optimizer over Bitcoin historical data.

##      Applied Concepts:
        Technical Analysis (TA) via Weighted Moving Averages (WMA)
        Fitness Evaluation: Calculates profit using historical price data by simulating trades triggered by moving average crossovers.
        Parameter Optimization: 14-dimensional vector optimization (weights, durations, smoothing factors for both low and high filters).

## aco.py – Ant Colony Optimization (ACO)
This file implements the Ant Colony Optimization (ACO) algorithm to optimize parameters for the Bitcoin trading bot. Inspired by ant foraging behavior, this algorithm adapts over iterations by reinforcing high-performing parameter combinations.

##      Applied Concepts::
        Ant Colony Optimization (ACO): Simulates collective learning through pheromone-based reinforcement.
        Pheromone Trails: Maintains and updates strength of parameter choices based on fitness performance.
        Exploration vs Exploitation: Balances exploration of new solutions with exploitation of previously successful ones.
        Parameter Sampling: Generates randomized 14-dimensional configurations (weights, durations, and smoothing factors) for each ant.

##      Key Functions:
        sample_parameters() – Generates candidate low and high filter parameters for each ant.
        run() – Executes multiple iterations where ants propose solutions, and the best-performing ones reinforce pheromone trails.
        Integration with Evaluator – Each solution is evaluated using the Evaluator class for historical trading performance.
        Final Evaluation – Tests the best parameter set on unseen (2020–2022) data to evaluate generalization.
