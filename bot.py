import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class EvaluationFunction:
    def __init__(self, csv_source, transaction_cost=0.03, start_date=None, end_date=None, starting_budget=1000):
        self.transaction_cost = transaction_cost
        self.money = starting_budget

        if isinstance(csv_source, str): df = pd.read_csv(csv_source)
        else: df = pd.concat([pd.read_csv(file) for file in csv_source])

        df.index = pd.to_datetime(df.pop("date"))
        df = df.sort_index().loc[start_date:end_date]

        self.df = df
        self.low_filter = None
        self.high_filter = None
    
    def set_filters(self, low_filter, high_filter):
        def sma(N):
            weights = np.ones(N)
            return weights / weights.sum()

        def lma(N):
            weights = np.linspace(1, N, N)
            return weights / weights.sum()
        
        def ema(N, alpha):
            weights = np.array([(1 - alpha) ** i for i in range(N)])
            return weights / weights.sum()
        
        def calculate_filter(filter_values):
            w1, w2, w3, d1, d2, d3, a3 = filter_values
            max_d = max(d1, d2, d3)

            filters = [
            w1 * np.pad(sma(d1), (0, max_d - d1), 'constant'),
            w2 * np.pad(lma(d2), (0, max_d - d2), 'constant'),
            w3 * np.pad(ema(d3, a3), (0, max_d - d3), 'constant')
            ]

            return sum(filters) / (w1 + w2 + w3)

        self.low_filter = calculate_filter(low_filter)
        self.high_filter = calculate_filter(high_filter)

        return [self.low_filter, self.high_filter]

    def _calculate_signals(self, df):
        def wma(P, kernel):
            padding = P[:len(kernel) - 1][::-1]
            P = np.concatenate((padding, P))

            return np.convolve(P, kernel, "valid")
    
        df["low"] = wma(df["close"], self.low_filter)
        df["high"] = wma(df["close"], self.high_filter)

        df["diff"] = df['high'] - df['low']
        df['sell_signal'] = (df['diff'].shift(1) < 0) & (df['diff'] >= 0)
        df['buy_signal'] = (df['diff'].shift(1) > 0) & (df['diff'] <= 0)

        return df

    def calculate_fitness(self):
        self.df = self._calculate_signals(self.df)

        buy_signals = self.df[self.df['buy_signal']]
        sell_signals = self.df[self.df['sell_signal']]

        money = self.money
        money -= (buy_signals["close"] + self.transaction_cost).sum()
        money += (sell_signals["close"] - self.transaction_cost).sum()

        return money
    
    def plot_filters(self):
        if not self.low_filter.all or not self.high_filter.all:
            raise ValueError("Low and High filters must be set before plotting.")
        
        plt.figure(figsize=(10, 5))

        plt.plot(self.low_filter, label="Low Filter", marker="o", linestyle="-", alpha=0.7)
        plt.plot(self.high_filter, label="High Filter", marker="s", linestyle="--", alpha=0.7)

        plt.title("Low and High Filters")
        plt.xlabel("Index")
        plt.ylabel("Filter Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def plot_signals(self):
        self.df = self._calculate_signals(self.df)

        plt.figure(figsize=(12, 6))

        plt.plot(self.df.index, self.df["close"], label="Close", linewidth=0.5)
        plt.plot(self.df.index, self.df["low"], label="Low", linestyle="--", alpha=0.5)
        plt.plot(self.df.index, self.df["high"], label="High", linestyle="--", alpha=0.5)

        plt.scatter(self.df.index[self.df['buy_signal']], self.df['close'][self.df['buy_signal']] + 2000, label="Buy Signal", marker="^", color="green", alpha=0.7, s=10)
        plt.scatter(self.df.index[self.df['sell_signal']], self.df['close'][self.df['sell_signal']] + 2000, label="Sell Signal", marker="v", color="red", alpha=0.7, s=10)

        plt.title("Bitcoin Stock Trading Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    daily = "data/BTC-Daily.csv"
    evaluator = EvaluationFunction(daily, end_date="2020-01-01")

    evaluator.set_filters([0.6, 0, 0.4, 10, 0, 10, 0.2], [1, 0, 0, 30, 0, 0, 0])
    evaluator.plot_filters()
    print(evaluator.calculate_fitness())
    evaluator.plot_signals()