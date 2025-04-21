#!/usr/bin/env python3
"""
AI Trading Bot - dual-mode edition
----------------------------------
mode="blend" : original 14-parameter HIGH/LOW weighted-average crossover
mode="macd"  : 3-parameter (optional 4) MACD strategy

Optimisers included:
    • PPSO  (Phasor Particle Swarm Optimisation)
    • HGSA  (Hybrid Genetic + Simulated Annealing)

Synthetic data generator is left in place; swap with real CSV loader for production.
"""
import numpy as np
import random, math
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Core moving-average helpers
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

# ---------------------------------------------------------------------------
# 1-bis.  MACD utilities
# ---------------------------------------------------------------------------
def compute_MACD(prices, short_span, long_span, signal_span):
    ema_short  = compute_EMA_full(prices, short_span)
    ema_long   = compute_EMA_full(prices, long_span)
    macd_line  = ema_short - ema_long
    signal_line = compute_EMA_full(macd_line, signal_span)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

# ---------------------------------------------------------------------------
# 2. Trading bot
# ---------------------------------------------------------------------------
class TradingBot:
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

# ---------------------------------------------------------------------------
# 3.  Evaluation wrapper
# ---------------------------------------------------------------------------
def evaluation_function(params, bot):
    bot.update_parameters(params)
    return bot.simulate_trading()

# ---------------------------------------------------------------------------
# 4. Optimiser base
# ---------------------------------------------------------------------------
class Optimizer:
    def __init__(self, pop_size, max_iter):
        self.pop_size = pop_size
        self.max_iter = max_iter

# ---------------------------------------------------------------------------
# 5. PPSO
# ---------------------------------------------------------------------------
class PPSO(Optimizer):
    def __init__(self, pop_size=40, max_iter=80):
        super().__init__(pop_size, max_iter)

    def initialize(self, dim, bounds):
        pop   = [[random.uniform(bounds[d][0], bounds[d][1])
                  for d in range(dim)] for _ in range(self.pop_size)]
        theta = [random.uniform(0, 2*math.pi) for _ in range(self.pop_size)]
        return pop, theta

    def optimize(self, bot, eval_fn, dim, bounds):
        pop, theta = self.initialize(dim, bounds)
        g_best, g_val = pop[0], -float('inf')

        for it in range(self.max_iter):
            for i in range(self.pop_size):
                # --- PPSO velocity update (phasor rule) ---
                c1 = abs(math.cos(theta[i]))**2 * math.sin(theta[i])
                c2 = abs(math.sin(theta[i]))**2 * math.cos(theta[i])
                r1, r2 = random.random(), random.random()
                v = [0]*dim
                p_best = pop[i]                      # no personal best memory for simplicity
                for d in range(dim):
                    v[d] = (c1*r1*(p_best[d] - pop[i][d]) +
                            c2*r2*(g_best[d] - pop[i][d]))
                    pop[i][d] += v[d]
                    pop[i][d] = max(bounds[d][0], min(pop[i][d], bounds[d][1]))
                # evaluate
                val = eval_fn(pop[i], bot)
                if val > g_val:
                    g_best, g_val = pop[i][:], val
                # phase angle drift
                theta[i] = (theta[i] + random.uniform(0, 2*math.pi)) % (2*math.pi)
            print(f"PPSO iter {it+1}/{self.max_iter}  best={g_val:.2f}")
        return g_best

# ---------------------------------------------------------------------------
# 6. HGSA  (simple version)
# ---------------------------------------------------------------------------
class HGSA(Optimizer):
    def __init__(self, pop_size=40, max_iter=80):
        super().__init__(pop_size, max_iter)

    def initialize(self, dim, bounds):
        return [[random.uniform(bounds[d][0], bounds[d][1])
                 for d in range(dim)] for _ in range(self.pop_size)]

    def optimize(self, bot, eval_fn, dim, bounds):
        pop = self.initialize(dim, bounds)
        temps = [1.0]*self.pop_size
        g_best, g_val = pop[0], -float('inf')

        for it in range(self.max_iter):
            scores = [eval_fn(ind, bot) for ind in pop]
            # update global best
            for ind, sc in zip(pop, scores):
                if sc > g_val:
                    g_best, g_val = ind[:], sc
            # GA-style evolution
            new_pop = []
            while len(new_pop) < self.pop_size:
                a, b = random.sample(pop, 2)
                cut = random.randint(1, dim-1)
                child = a[:cut] + b[cut:]
                # mutation
                for d in range(dim):
                    if random.random() < 0.1:
                        perturb = random.uniform(-1, 1)*(bounds[d][1]-bounds[d][0])*0.1
                        child[d] = max(bounds[d][0], min(child[d]+perturb, bounds[d][1]))
                new_pop.append(child)
            pop = new_pop
            # SA-style refinement
            for i in range(self.pop_size):
                cand = pop[i][:]
                for d in range(dim):
                    cand[d] += np.random.normal(0, temps[i]) * (bounds[d][1]-bounds[d][0])*0.05
                    cand[d] = max(bounds[d][0], min(cand[d], bounds[d][1]))
                if eval_fn(cand, bot) > eval_fn(pop[i], bot):
                    pop[i] = cand
            temps = [t*0.95 for t in temps]       # cool
            print(f"HGSA iter {it+1}/{self.max_iter}  best={g_val:.2f}")
        return g_best

# ---------------------------------------------------------------------------
# 7. Demo data loader (replace with CSV)
# ---------------------------------------------------------------------------
def load_data():
    t = np.linspace(0, 12*np.pi, 1200)
    prices = 100 + 10*np.sin(t) + np.random.normal(0,2,len(t))
    return prices.tolist()

# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prices = load_data()

    # -------- MACD mode demo --------------------------------------------
    bot_macd = TradingBot(prices, mode="macd")
    dim_macd = 3
    bounds_macd = [(5,50), (20,200), (2,30)]      # short, long, signal

    pso = PPSO(pop_size=30, max_iter=60)
    best_macd = pso.optimize(bot_macd, evaluation_function,
                             dim_macd, bounds_macd)
    print("\nBest MACD params:", best_macd)
    print("Final cash:", evaluation_function(best_macd, bot_macd))

    # -------- Blend mode demo -------------------------------------------
    bot_blend = TradingBot(prices, mode="blend")
    dim_blend = 14
    # (w,w,w, d,d,d, α) ×2
    b = []
    b.extend([(0,100)]*3 + [(5,100)]*3 + [(0.05,0.95)])
    b.extend([(0,100)]*3 + [(5,100)]*3 + [(0.05,0.95)])
    bounds_blend = b

    ga = HGSA(pop_size=30, max_iter=60)
    best_blend = ga.optimize(bot_blend, evaluation_function,
                             dim_blend, bounds_blend)
    print("\nBest BLEND params:", best_blend)
    print("Final cash:", evaluation_function(best_blend, bot_blend))

    # -------- Visual quick-look -----------------------------------------
    plt.figure(figsize=(10,3))
    plt.plot(prices)
    plt.title("Synthetic price series"); plt.show()


