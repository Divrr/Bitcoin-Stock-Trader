import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import time  # for evaluation timing
from config import DATA_CFG 

# --- Weighted Moving Average kernels ---
# Simple Moving Average kernel: equal weights over window of length N
# Normalised so weights sum to 1

def sma_kernel(N):
    weights = np.array([1 for _ in range(N)])
    return weights / N

# Linear-weighted Moving Average kernel: linearly decaying weights
# Normalised by 2/(N+1) so weights sum to 1

def lma_kernel(N):
    weights = np.array([1 - k/N for k in range(N)])
    return weights * (2/(N+1))

# Exponential Moving Average kernel: exponentially decaying weights
# Alpha controls decay rate. Only first N terms used.

def ema_kernel(N, alpha):
    weights = np.array([(1 - alpha) ** k for k in range(N)])
    return weights * alpha

# Convolution helper for any WMA filter
# Pads input series by reflecting first window values to avoid edge effects

def wma(P, kernel):
    padding = -np.flip(P[1:len(kernel)]-P[0]) + P[0]  # flip and negate for padding
    P_padded = np.concatenate((padding, P))
    return np.convolve(P_padded, kernel, mode="valid")

# Compute MACD line, signal line, and histogram given price array

def compute_MACD(prices, short_span, long_span, signal_span):
    ema_short = wma(prices, ema_kernel(short_span, alpha=2/(short_span+1)))
    ema_long  = wma(prices, ema_kernel(long_span,  alpha=2/(long_span+1)))
    macd_line = ema_short - ema_long
    signal_line = wma(macd_line, ema_kernel(signal_span, alpha=2/(signal_span+1)))
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

class Evaluator:
    def __init__(self, prices, mode=None):
        """
        Initialize evaluator with price series and mode.
        mode="blend" uses 14-parameter high/low WMA crossover
        mode="macd" uses MACD strategy (3-4 parameters)
        """
        self.prices = np.array(prices, dtype=float)
        self.df = pd.DataFrame(prices)
        self.visualise = False
        self.mode   = mode if mode is not None else DATA_CFG.get("mode")
        self.params = None

        # Evaluation cost metrics
        self.eval_count = 0   # total number of fitness calls
        self.eval_time  = 0.0 # total time spent in simulate_trading (seconds)

    # ---------------------------------------------------------------------
    def update_parameters(self, params):
        """
        Validate and store the parameter vector for the chosen mode.
        """
        if self.mode == "blend" and len(params) != 14:
            raise ValueError("blend mode expects 14 parameters")
        if self.mode == "macd" and len(params) not in (3, 4):
            raise ValueError("macd mode expects 3 or 4 parameters")
        if self.mode == "2d_sma" and len(params) != 2:
            raise ValueError("2d_sma mode expects 2 parameters")
        if self.mode == "21d_macd" and len(params) != 21:
            raise ValueError("21d_macd mode expects 21 parameters")
        self.params = params

    # ---------------------------------------------------------------------
    def evaluate(self, params):
        """
        Perform one fitness evaluation:
          1. Update parameters
          2. Time the simulate_trading() call
          3. Increment eval_count and accumulate eval_time
        Returns the final cash balance as fitness.
        """
        self.update_parameters(params)
        start = time.time()
        fitness = self.simulate_trading()
        elapsed = time.time() - start
        self.eval_time  += elapsed
        self.eval_count += 1
        return fitness

    # ---------------------------------------------------------------------
    def simulate_trading(self):
        """
        Dispatch to the correct simulation based on mode.
        """
        result = None
        if self.params is None:
            raise ValueError("Parameters not set")
        if self.mode == "blend":
            result = self._simulate_blend()
        elif self.mode == "macd":
            result = self._simulate_macd()
        elif self.mode == "2d_sma":
            result = self._simulate_2d_sma()
        elif self.mode == "21d_macd":
            result = self._simulate_21d_macd()
        else:
            raise ValueError("Unknown mode")
        
        if self.visualise and self.mode == "blend":
            # unfortunately this is the only one I could build in the time I had
            self.plot() 
        
        return result

    # ---- original weighted-average crossover ----------------------------
    def _simulate_blend(self):
        p = self.params
        hi, lo = p[:7], p[7:]

        # unpack weights and durations
        w1h, w2h, w3h, d1h, d2h, d3h, ah = hi
        w1l, w2l, w3l, d1l, d2l, d3l, al = lo

        # enforce minimum window size of 5 and integer durations
        d1h, d2h, d3h = [max(5, int(round(x))) for x in (d1h, d2h, d3h)]
        d1l, d2l, d3l = [max(5, int(round(x))) for x in (d1l, d2l, d3l)]

        # compute component WMAs
        sma_h = wma(self.prices, sma_kernel(d1h))
        lma_h = wma(self.prices, lma_kernel(d2h))
        ema_h = wma(self.prices, ema_kernel(d3h, ah))
        sma_l = wma(self.prices, sma_kernel(d1l))
        lma_l = wma(self.prices, lma_kernel(d2l))
        ema_l = wma(self.prices, ema_kernel(d3l, al))

        # combine into high vs low signals then generate binary position
        high = (w1h*sma_h + w2h*lma_h + w3h*ema_h) / (w1h + w2h + w3h + 1e-6)
        low  = (w1l*sma_l + w2l*lma_l + w3l*ema_l) / (w1l + w2l + w3l + 1e-6)
        signal = np.where(high > low, 1, -1)

        if self.visualise:
            self.df["high"] = high
            self.df["low"] = low
            self.df["signal"] = signal

        # backtest returns final cash amount
        return self._backtest(signal)

    # ---- MACD strategy --------------------------------------------------
    def _simulate_macd(self):
        # interpret parameters with safe bounds
        short_span  = max(5, int(round(self.params[0])))
        long_span   = max(short_span + 1, int(round(self.params[1])))
        signal_span = max(2, int(round(self.params[2])))
        threshold   = self.params[3] if len(self.params) == 4 else 0.0

        macd, sig_line, hist = compute_MACD(
            self.prices, short_span, long_span, signal_span
        )

        # detect zero crossings for buy/sell
        bullish = (hist[1:] > threshold) & (hist[:-1] <= threshold)
        bearish = (hist[1:] <= threshold) & (hist[:-1] >  threshold)
        signal = np.zeros_like(self.prices, dtype=int)
        signal[1:][bullish] = 1
        signal[1:][bearish] = -1

        # hold position until opposite signal appears
        for i in range(1, len(signal)):
            if signal[i] == 0:
                signal[i] = signal[i-1] if signal[i-1] != 0 else -1

        return self._backtest(signal)

    # ========== 2D SMA mode ==========
    def _simulate_2d_sma(self):
        short, long = [max(2, int(round(x))) for x in self.params]
        sma_short = wma(self.prices, sma_kernel(short))
        sma_long  = wma(self.prices, sma_kernel(long))
        signal = np.where(sma_short > sma_long, 1, -1)
        return self._backtest(signal)
    

    # ==========  21D MACD mode ==========
    def _simulate_21d_macd(self):
        p = self.params
        fast7   = p[:7]
        slow7   = p[7:14]
        sig7    = p[14:21]
        # unpack weights and durations for each filter
        def build_kernel(w1, w2, w3, d1, d2, d3, a):
            d1, d2, d3 = [max(2, int(round(x))) for x in (d1, d2, d3)]
            k1 = w1 * sma_kernel(d1)
            k2 = w2 * lma_kernel(d2)
            k3 = w3 * ema_kernel(d3, a)
            total_w = w1 + w2 + w3 + 1e-6
            maxlen = max(len(k1), len(k2), len(k3))
            def pad(k, L): return np.pad(k, (0, L - len(k)), 'constant')
            return (pad(k1, maxlen) + pad(k2, maxlen) + pad(k3, maxlen)) / total_w

        fast_kernel = build_kernel(*fast7)
        slow_kernel = build_kernel(*slow7)
        sig_kernel  = build_kernel(*sig7)
        fast_line = wma(self.prices, fast_kernel)
        slow_line = wma(self.prices, slow_kernel)
        macd_line = fast_line[-len(slow_line):] - slow_line
        sig_line = wma(macd_line, sig_kernel)
        hist = macd_line[-len(sig_line):] - sig_line
        bullish = (hist[1:] > 0) & (hist[:-1] <= 0)
        bearish = (hist[1:] <= 0) & (hist[:-1] > 0)
        signal = np.zeros_like(self.prices, dtype=int)
        minlen = len(signal) - len(hist)
        signal[-len(hist)+1:][bullish] = 1
        signal[-len(hist)+1:][bearish] = -1
        # Position signal shift
        for i in range(1, len(signal)):
            if signal[i] == 0:
                signal[i] = signal[i-1] if signal[i-1] != 0 else -1
        return self._backtest(signal)


    # ---- common back-tester --------------------------------------------
    def _backtest(self, signal):
        # starting capital and no holdings
        cash, btc = 1000.0, 0.0
        fee = 0.03
        cash_history = []
        
        for idx, price in enumerate(self.prices):
            if signal[idx] == 1 and cash > 0:
                # buy full position
                btc  = (cash * (1 - fee)) / price
                cash = 0.0
            elif signal[idx] == -1 and btc > 0:
                # sell full position
                cash = btc * price * (1 - fee)
                btc  = 0.0
            cash_history.append(cash + btc*price)
        # liquidate any remaining BTC at final price
        if btc > 0:
            cash = btc * self.prices[-1] * (1 - fee)

        if self.visualise:
            self.df["cash"] = cash_history
        
        return cash
    
    def plot(self):
        # ONLY WORKS FOR BLEND MODE
        signal = self.df["signal"]

        prev_signal = signal.shift(1)
        self.df["buy_signal"] = (prev_signal == -1) & (signal == 1)
        self.df["sell_signal"] = (prev_signal == 1) & (signal == -1)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(self.df.index, self.df["close"], label="Close", linewidth=0.5)
        ax1.plot(self.df.index, self.df["low"], label="Low", linestyle="--", alpha=0.5)
        ax1.plot(self.df.index, self.df["high"], label="High", linestyle="--", alpha=0.5)

        ax1.scatter(self.df.index[self.df['buy_signal']], self.df['close'][self.df['buy_signal']] + 2000, label="Buy Signal", marker="^", color="green", alpha=0.7, s=10)
        ax1.scatter(self.df.index[self.df['sell_signal']], self.df['close'][self.df['sell_signal']] + 2000, label="Sell Signal", marker="v", color="red", alpha=0.7, s=10)

        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(self.df.index, self.df["cash"], label="Cash", color="purple", linewidth=0.5)
        ax2.set_ylabel("Cash", color="purple")
        ax2.tick_params(axis='y', labelcolor='purple')

        plt.legend()
        plt.title("High/Low/Signal over Time")
        plt.show()