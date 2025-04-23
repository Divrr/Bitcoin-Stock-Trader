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

# ---------------------------------------------------------------------------
# 1. WMA helpers
# ---------------------------------------------------------------------------
def compute_SMA(prices, window):
    if window < 1: raise ValueError("window must be ≥1")
    sma = np.convolve(prices, np.ones(window) / window, mode="same")
    sma[:window-1] = sma[window-1]          # pad left side
    return sma

def compute_LMA(prices, window):
    if window < 1: raise ValueError("window must be ≥1")
    w = np.arange(1, window + 1)
    w = (w / w.sum())[::-1]                 # recent weight highest
    lma = np.convolve(prices, w, mode="same")
    lma[:window-1] = lma[window-1]
    return lma

def compute_EMA_full(prices, span):
    alpha = 2.0 / (span + 1.0)
    ema = np.zeros_like(prices, dtype=float)
    ema[span-1] = np.mean(prices[:span])
    ema[:span-1] = ema[span-1]              # fill warm-up
    for i in range(span, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

def compute_MACD(prices, short_span, long_span, signal_span):
    ema_short  = compute_EMA_full(prices, short_span)
    ema_long   = compute_EMA_full(prices, long_span)
    macd_line  = ema_short - ema_long
    signal_line = compute_EMA_full(macd_line, signal_span)
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

        sma_h = compute_SMA(self.prices, d1h)
        lma_h = compute_LMA(self.prices, d2h)
        ema_h = compute_EMA_full(self.prices, d3h) * ah + 0  # scale later

        sma_l = compute_SMA(self.prices, d1l)
        lma_l = compute_LMA(self.prices, d2l)
        ema_l = compute_EMA_full(self.prices, d3l) * al + 0

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