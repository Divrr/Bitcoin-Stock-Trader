# Bitcoin-Stock-Trader
CITS4404 project


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



# pip install dash plotly pandas matplotlib
