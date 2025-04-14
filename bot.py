import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()

df17 = pd.read_csv("data/BTC-2017min.csv")
df18 = pd.read_csv("data/BTC-2018min.csv")
df19 = pd.read_csv("data/BTC-2019min.csv")
df = pd.concat([df17, df18, df19])

df.index = pd.to_datetime(df.pop("date"))

#--------------------------------------
def pad(P,N):
    padding = np.flip(P[1:N])
    return np.append(padding, P)

def wma(P,N,kernel):
    return np.convolve(pad(P,N), kernel, "valid")
# source: bot.pdf
#--------------------------------------
# FILTERS
def sma_filter(N):
    return np.ones(N)/N
#--------------------------------------
# BOT
start_date = "2017-01-01"
end_date = "2020-01-01"
buy_sell_cost = 0
money = 0
low_sma = 2000
high_sma = 4000

df = df.sort_index().loc[start_date:end_date]
df["low"] = wma(df["close"], low_sma, sma_filter(low_sma))
df["high"] = wma(df["close"], high_sma, sma_filter(high_sma))

df["diff"] = df['high'] - df['low']
df['sell_signal'] = (df['diff'].shift(1) < 0) & (df['diff'] >= 0)
df['buy_signal'] = (df['diff'].shift(1) > 0) & (df['diff'] <= 0)

buy_signals = df[df['buy_signal']]
sell_signals = df[df['sell_signal']]

money -= (buy_signals["close"] + buy_sell_cost).sum()
money += (sell_signals["close"] - buy_sell_cost).sum()

print("You have made: $" + str(money))

#--------------------------------------
# PLOT
plt.figure(figsize=(12, 6))

plt.plot(df.index, df["close"], label="Close", linewidth=0.5)
plt.plot(df.index, df["low"], label="Low", linestyle="--", alpha=0.5)
plt.plot(df.index, df["high"], label="High", linestyle="--", alpha=0.5)

plt.scatter(df.index[df['buy_signal']], df['close'][df['buy_signal']] + 2000, label="Buy Signal", marker="^", color="green", alpha=0.7, s=10)
plt.scatter(df.index[df['sell_signal']], df['close'][df['sell_signal']] + 2000, label="Sell Signal", marker="v", color="red", alpha=0.7, s=10)

plt.title("Bitcoin Stock Trading Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

end=time.time()
print(end-start)

plt.show()