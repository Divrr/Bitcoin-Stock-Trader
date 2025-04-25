#!/usr/bin/env python3
"""
AI Trading Bot - dual-mode edition
----------------------------------
mode="blend" : original 14-parameter HIGH/LOW weighted-average crossover
mode="macd"  : 3-parameter (optional 4) MACD strategy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sma_kernel(N):
    weights = np.array([1 for k in range(N)])
    return weights / N

def lma_kernel(N):
    weights = np.array([1-k/N for k in range(N)])
    return weights * (2/(N+1))

def ema_kernel(N, alpha):
    weights = np.array([(1 - alpha) ** k for k in range(N)])
    return weights * alpha

def wma(P, kernel):
    padding = -P[:len(kernel) - 1][::-1]
    P = np.concatenate((padding, P))
    return np.convolve(P, kernel, "valid")

def compute_MACD(prices, short_span, long_span, signal_span):
    ema_short  = wma(prices, ema_kernel(short_span))
    ema_long   = wma(prices, ema_kernel(long_span))
    macd_line  = ema_short - ema_long
    signal_line = wma(macd_line, ema_kernel(signal_span))
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

class Evaluator:
    def __init__(self, prices, mode="blend"):
        """
        mode = "blend" (14 params) or "macd" (3-4 params)
        """
        self.prices = np.array(prices, dtype=float)
        self.mode   = mode
        self.params = None

    # ---------------------------------------------------------------------
    def update_parameters(self, params):
        if self.mode == "blend" and len(params) != 14:
            raise ValueError("blend mode expects 14 parameters")
        if self.mode == "macd" and len(params) not in (3, 4):
            raise ValueError("macd mode expects 3 or 4 parameters")
        self.params = params

    # ---------------------------------------------------------------------
    def evaluate(self, params):
        self.update_parameters(params)
        return self.simulate_trading()

    # ---------------------------------------------------------------------
    def simulate_trading(self):
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

        # unpack + coerce
        w1h, w2h, w3h, d1h, d2h, d3h, ah = hi
        w1l, w2l, w3l, d1l, d2l, d3l, al = lo
        d1h, d2h, d3h = [max(5, int(round(x))) for x in (d1h, d2h, d3h)]
        d1l, d2l, d3l = [max(5, int(round(x))) for x in (d1l, d2l, d3l)]

        sma_h = wma(self.prices, sma_kernel(d1h))
        lma_h = wma(self.prices, lma_kernel(d2h))
        ema_h = wma(self.prices, ema_kernel(d3h, ah))

        sma_l = wma(self.prices, sma_kernel(d1l))
        lma_l = wma(self.prices, lma_kernel(d2l))
        ema_l = wma(self.prices, ema_kernel(d3l, al))

        high = (w1h*sma_h + w2h*lma_h + w3h*ema_h) / (w1h+w2h+w3h+1e-6)
        low  = (w1l*sma_l + w2l*lma_l + w3l*ema_l) / (w1l+w2l+w3l+1e-6)
        signal = np.where(high > low, 1, -1)

        return self._backtest(signal)

    # ---- MACD strategy --------------------------------------------------
    def _simulate_macd(self):
        short_span  = max(5, int(round(self.params[0])))
        long_span   = max(short_span+1, int(round(self.params[1])))
        signal_span = max(2, int(round(self.params[2])))
        threshold   = self.params[3] if len(self.params) == 4 else 0.0

        macd, sig, hist = compute_MACD(self.prices,
                                       short_span, long_span, signal_span)

        bullish = (hist[1:] >  threshold) & (hist[:-1] <= threshold)
        bearish = (hist[1:] <= threshold) & (hist[:-1] >  threshold)
        signal = np.zeros_like(self.prices, dtype=int)
        signal[1:][bullish] = 1
        signal[1:][bearish] = -1
        # forward-fill: stay long until bearish appears, etc.
        for i in range(1, len(signal)):
            if signal[i] == 0:
                signal[i] = signal[i-1] if signal[i-1] != 0 else -1

        return self._backtest(signal)

    # ---- common back-tester --------------------------------------------
    def _backtest(self, signal):
        cash, btc = 1000.0, 0.0
        fee = 0.03
        for i in range(len(self.prices)):
            if signal[i] == 1 and cash > 0:
                btc  = (cash * (1-fee)) / self.prices[i]
                cash = 0.0
            elif signal[i] == -1 and btc > 0:
                cash = btc * self.prices[i] * (1-fee)
                btc  = 0.0
        if btc > 0:
            cash = btc * self.prices[-1] * (1-fee)
        return cash