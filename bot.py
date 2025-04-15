import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class EvaluationFunction:
    def __init__(self, csv_source, transaction_cost=0, start_date=None, end_date=None, starting_budget=0):
        self.transaction_cost = transaction_cost
        self.money = starting_budget

        df = pd.concat([pd.read_csv(file) for file in csv_source])
        df.index = pd.to_datetime(df.pop("date"))
        df = df.sort_index().loc[start_date:end_date]

        self.df = df
        self.model_parameters = np.zeros(2, dtype=float)

    def _calculate_signals(self, df, low_sma, high_sma):
        def pad(P,N):
            padding = np.flip(P[1:N])
            return np.append(padding, P)

        def wma(P,N,kernel):
            return np.convolve(pad(P,N), kernel, "valid")
        
        def sma_filter(N):
            return np.ones(N)/N
    
        df["low"] = wma(df["close"], low_sma, sma_filter(low_sma))
        df["high"] = wma(df["close"], high_sma, sma_filter(high_sma))

        df["diff"] = df['high'] - df['low']
        df['sell_signal'] = (df['diff'].shift(1) < 0) & (df['diff'] >= 0)
        df['buy_signal'] = (df['diff'].shift(1) > 0) & (df['diff'] <= 0)

        return df

    def calculate_fitness(self):
        low_sma, high_sma = self.model_parameters
        self.df = self._calculate_signals(self.df, low_sma, high_sma)

        buy_signals = self.df[self.df['buy_signal']]
        sell_signals = self.df[self.df['sell_signal']]

        money = self.money
        money -= (buy_signals["close"] + self.transaction_cost).sum()
        money += (sell_signals["close"] - self.transaction_cost).sum()

        return money

    def plot_signals(self):
        low_sma, high_sma = self.model_parameters
        self.df = self._calculate_signals(self.df, low_sma, high_sma)

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
    files = ["data/BTC-Daily.csv"]
    evaluator = EvaluationFunction(files, transaction_cost=10, end_date="2020-01-01")
    evaluator.model_parameters = [10, 20]

    profit = evaluator.calculate_fitness()
    print(f"Profit: {profit}")

    evaluator.plot_signals()