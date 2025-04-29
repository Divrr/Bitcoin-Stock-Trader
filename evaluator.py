#!/usr/bin/env python3
"""
AI Trading Bot - dual-mode edition with evaluation cost tracking
----------------------------------
mode="blend" : original 14-parameter HIGH/LOW weighted-average crossover
mode="macd"  : 3-parameter (optional 4) MACD strategy
"""

import numpy as np
import time  # for evaluation timing

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
    padding = -P[:len(kernel) - 1][::-1]  # flip and negate for padding
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
    def __init__(self, prices, mode="blend"):
        """
        Initialize evaluator with price series and mode.
        mode="blend" uses 14-parameter high/low WMA crossover
        mode="macd" uses MACD strategy (3-4 parameters)
        """
        self.prices = np.array(prices, dtype=float)
        self.mode   = mode
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
        if self.params is None:
            raise ValueError("Parameters not set")
        if self.mode == "blend":
            return self._simulate_blend()
        else:
            return self._simulate_macd()

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

    # ---- common back-tester --------------------------------------------
    def _backtest(self, signal):
        # starting capital and no holdings
        cash, btc = 1000.0, 0.0
        fee = 0.03
        
        for idx, price in enumerate(self.prices):
            if signal[idx] == 1 and cash > 0:
                # buy full position
                btc  = (cash * (1 - fee)) / price
                cash = 0.0
            elif signal[idx] == -1 and btc > 0:
                # sell full position
                cash = btc * price * (1 - fee)
                btc  = 0.0
        # liquidate any remaining BTC at final price
        if btc > 0:
            cash = btc * self.prices[-1] * (1 - fee)
        return cash
