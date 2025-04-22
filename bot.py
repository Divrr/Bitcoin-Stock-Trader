import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless environments

class Evaluator:
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
            weights = np.array([1 for k in range(N)])
            return weights * N

        def lma(N):
            weights = np.array([1-k/N for k in range(N)])
            return weights * (2/(N+1))
        
        def ema(N, alpha):
            weights = np.array([(1 - alpha) ** k for k in range(N)])
            return weights * alpha
        
        def calculate_filter(filter_values):
            w1, w2, w3 = filter_values[:3]
            d1, d2, d3 = map(int, map(round, filter_values[3:6]))
            a3 = float(filter_values[6])
            
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
    
    def plot_filters(self):
        if not self.low_filter.all or not self.high_filter.all:
            raise ValueError("Low and High filters must be set before plotting.")
        
        plt.figure(figsize=(10, 5))

        plt.plot(self.low_filter, label="Low Filter", marker="o", linestyle="-", alpha=0.7)
        plt.plot(self.high_filter, label="High Filter", marker="s", linestyle="--", alpha=0.7)

        plt.title("Low and High Filters")
        plt.xlabel("Index")
        plt.ylabel("Filter Value")
        plt.ylim(ymin=0)
        plt.grid()
        plt.legend()
        plt.show()

    def generate_signals(self):
        def wma(P, kernel):
            padding = P[:len(kernel) - 1][::-1]
            P = np.concatenate((padding, P))
            return np.convolve(P, kernel, "valid")
        
        df = self.df 
    
        df["low"] = wma(df["close"], self.low_filter)
        df["high"] = wma(df["close"], self.high_filter)

        df["diff"] = df['high'] - df['low']
        df['sell_signal'] = (df['diff'].shift(1) < 0) & (df['diff'] >= 0)
        df['buy_signal'] = (df['diff'].shift(1) > 0) & (df['diff'] <= 0)

        return df

    def calculate_fitness(self):
        cash = self.money
        bitcoin = 0
        cash_column = []

        for index, row in self.df.iterrows():
            if row['buy_signal'] and cash > 0:
                cash -= cash * self.transaction_cost
                bitcoin = cash / row['close']
                cash = 0
            elif row['sell_signal'] and bitcoin > 0:
                cash = bitcoin * row['close']
                cash -= cash * self.transaction_cost
                bitcoin = 0
            cash_column.append((cash + (bitcoin * row['close'])))

        self.df['profit'] = cash_column
        
        return cash_column[-1]

    def plot_signals(self, save_path=None):

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df["close"], label="Close", linewidth=0.5)
        plt.plot(self.df.index, self.df["low"], label="Low", linestyle="--", alpha=0.5)
        plt.plot(self.df.index, self.df["high"], label="High", linestyle="--", alpha=0.5)

        plt.scatter(self.df.index[self.df['buy_signal']], self.df['close'][self.df['buy_signal']] + 2000,
                    label="Buy Signal", marker="^", color="green", alpha=0.7, s=10)
        plt.scatter(self.df.index[self.df['sell_signal']], self.df['close'][self.df['sell_signal']] + 2000,
                    label="Sell Signal", marker="v", color="red", alpha=0.7, s=10)

        plt.title("Bitcoin Trading Signals and Filters")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc="upper left")
        plt.grid()
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path)
            print("✅ Plot saved to", save_path)
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    daily = "data/BTC-Daily.csv"

    evaluator = Evaluator(daily, start_date="2017-01-01", end_date="2019-12-31")
    evaluator.set_filters([1, 0, 1, 10, 0, 10, 0.5], [1, 0, 0, 20, 0, 0, 0])
    evaluator.plot_filters()
    
    evaluator.generate_signals()
    evaluator.plot_signals()
    
    profit = evaluator.calculate_fitness()
    print(profit)