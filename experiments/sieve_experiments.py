"""
sieve_experiments.py - stand-alone tests of the Pi-Phi "fixed" sieve (golden-ratio position filter).
One file, numpy only.  Uses sieve_lab.py (same folder).

    python sieve_experiments.py            run all experiments
    python sieve_experiments.py --test     self-checks; prints "all N checks passed"

Every experiment compares the golden sieve with plain alternatives: stride, random, stratified-random
(one random pick per block), and other irrational rotations.  Seeds are fixed.  Nothing here is tuned.
"""
from __future__ import annotations
import math, sys
import numpy as np
from sieve_lab import PHI, sieve_channels, alias_score, max_gap, stride_keep, random_keep

# ---------------------------------------------------------------- the schemes (all take n, rate -> bool mask)
def rot_keep(n, r, alpha):
    pos = np.arange(1, n + 1, dtype=np.float64)
    return ((pos * alpha) % 1.0) < r

def golden(n, r):   return rot_keep(n, r, PHI)
def sqrt2(n, r):    return rot_keep(n, r, math.sqrt(2))
def pi_rot(n, r):   return rot_keep(n, r, math.pi)
def stride(n, r):   return stride_keep(n, int(round(1 / r)), 0)
def rand(n, r, seed=0): return random_keep(n, r, np.random.default_rng(seed))
def strat(n, r, seed=0):
    """one random pick per block of k tokens (k = 1/r)"""
    k = int(round(1 / r)); rng = np.random.default_rng(seed); m = np.zeros(n, bool)
    for s in range(0, n, k):
        m[s + rng.integers(0, min(k, n - s))] = True
    return m

def window_dev(keep, r, W):
    """worst |kept count - r*W| over every window of W consecutive tokens"""
    c = np.concatenate([[0], np.cumsum(keep)])
    return float(np.abs(c[W:] - c[:-W] - r * W).max())

def exact_channel(n: int) -> int:
    """exact 3-way channel of the fixed sieve for ANY integer n >= 1 (integer arithmetic only)."""
    return (3 * n + math.isqrt(45 * n * n)) // 2 - 3 * ((n + math.isqrt(5 * n * n)) // 2)

# ---------------------------------------------------------------- experiments
def e1_rates(n=100_000):
    print("E1  other rates and other irrationals  (n=%d)" % n)
    print(f"{'rate':>6} {'scheme':<9}{'realized':>9}{'maxgap':>7}{'alias2-64':>10}{'win30 dev':>10}{'win300 dev':>11}")
    for r in [1/2, 1/3, 1/4, 1/5, 1/7, 1/10]:
        for name, f in [("golden", golden), ("sqrt2", sqrt2), ("pi", pi_rot), ("stride", stride), ("random", rand), ("strat", strat)]:
            m = f(n, r)
            al = alias_score(m, periods=range(2, 65))
            print(f"{r:>6.3f} {name:<9}{m.mean():>9.4f}{max_gap(m):>7}{al:>10.3f}{window_dev(m, r, 30):>10.2f}{window_dev(m, r, 300):>11.2f}")
        print()

def e2_periodic(N=3000, trials=40):
    print("E2  estimating the mean of periodic data from the kept third  (N=%d, periods 2..60, random phase)" % N)
    rng = np.random.default_rng(1)
    pos = np.arange(1, N + 1)
    out = {}
    for name, f in [("golden", golden), ("stride", stride), ("random", rand), ("strat", strat)]:
        errs = []                              # error per period (worst over phases)
        for p in range(2, 61):
            worst = 0.0
            for _ in range(trials // 4):
                th = rng.uniform(0, 2 * np.pi)
                g = 1 + np.cos(2 * np.pi * pos / p + th)     # true mean ~1
                m = f(N, 1/3)
                worst = max(worst, abs(g[m].mean() - g.mean()))
            errs.append(worst)
        errs = np.array(errs); out[name] = errs
        bad = [p + 2 for p in np.flatnonzero(errs > 0.2)]
        print(f"  {name:<8} median err {np.median(errs):.3f}   worst err {errs.max():.3f}   periods with err>0.2: {bad[:12]}{'...' if len(bad) > 12 else ''}")
    print()

def e3_folds(N=2999):
    print("E3  3-way split (folds) of time-ordered data: spread of fold means (max-min), lower = fairer  (N=%d)" % N)
    pos = np.arange(1, N + 1)
    sigs = {"trend only": pos / N, "period 3": np.cos(2*np.pi*pos/3), "period 6": np.cos(2*np.pi*pos/6 + 1),
            "period 7 (weekly)": np.cos(2*np.pi*pos/7 + 2), "period 24 (daily)": np.cos(2*np.pi*pos/24),
            "trend+period 12": pos / N + np.cos(2*np.pi*pos/12 + 0.5)}
    def folds(name):
        if name == "golden":  return sieve_channels(N, "fixed")
        if name == "mod 3":   return pos % 3
        if name == "blocks":  return np.minimum((pos - 1) * 3 // N, 2)
        if name == "random":  return np.random.default_rng(3).integers(0, 3, N)
        if name == "strat":
            rng = np.random.default_rng(4); ch = np.zeros(N, int)
            for s in range(0, N, 3):
                b = min(3, N - s); ch[s:s + b] = rng.permutation(3)[:b]
            return ch
    names = ["golden", "mod 3", "blocks", "random", "strat"]
    print(f"{'signal':<20}" + "".join(f"{n:>9}" for n in names))
    for sn, s in sigs.items():
        row = []
        for nm in names:
            ch = folds(nm); mu = [s[ch == j].mean() for j in range(3)]
            row.append(max(mu) - min(mu))
        print(f"{sn:<20}" + "".join(f"{v:>9.3f}" for v in row))
    print()

def e4_threeGap_and_prefix():
    print("E4  structure: gaps per channel, prefix stability")
    ch = sieve_channels(200_000, "fixed")
    for j in range(3):
        idx = np.flatnonzero(ch == j); g = np.diff(idx)
        print(f"  channel {j}: share {len(idx)/len(ch):.4f}   distinct gap sizes {sorted(set(g.tolist()))}   max gap {g.max()}")
    ok = all(np.array_equal(sieve_channels(k, "fixed"), ch[:k]) for k in (10, 1000, 77_777))
    print(f"  channel of token n does not depend on total length: {ok}")
    # shortest windows with all three channels present
    w = 1
    while not all(len(set(ch[s:s + w].tolist())) == 3 for s in range(0, 5000)): w += 1
    print(f"  every window of {w} consecutive tokens contains all 3 channels\n")

def e5_precision(samples=4000):
    print("E5  float64 vs exact integer arithmetic (channel of token n, mismatch rate)")
    rng = np.random.default_rng(5)
    print(f"{'n around':>10} {'mismatch':>10}")
    for k in [3, 6, 9, 12, 14, 15, 16]:
        ns = [int(10 ** k) + int(x) for x in rng.integers(0, 10 ** min(k, 9), samples)]
        bad = 0
        for n in ns:
            fl = int(math.floor(3 * ((n / PHI) % 1.0)))
            bad += fl != exact_channel(n)
        print(f"{'1e%d' % k:>10} {bad / samples:>10.4f}")
    print()

def hybrid_floor(scores, rate, floor_mask):
    keep = floor_mask.copy()
    extra = int(round(rate * len(scores))) - int(keep.sum())
    if extra > 0:
        s = np.where(keep, -np.inf, scores); keep[np.argsort(-s)[:extra]] = True
    return keep

def e6_hybrid(n=3000, trials=150, rate=1/3, floor_share=0.5, detect=0.5):
    print("E6  hybrid = floor + content scorer: does the GOLDEN floor beat other floors?")
    print(f"    n={n}, budget {rate:.2f}, floor share {floor_share}; scorer sees only {int(detect*100)}% of important tokens (the rest are 'hidden needles')")
    fr = rate * floor_share
    rng = np.random.default_rng(6); pos = np.arange(1, n + 1)
    layouts = {"random placement": lambda: rng.random(n) < 0.05,
               "periodic (every 5th)": lambda: (pos % 5) == 4,
               "periodic (every 6th)": lambda: (pos % 6) == 0,
               "bursts of 8": lambda: np.repeat(rng.random(n // 8 + 1) < 0.05, 8)[:n]}
    floors = {"golden": lambda: rot_keep(n, fr, PHI), "stride": lambda: stride(n, fr), "random": lambda: rand(n, fr, int(rng.integers(1 << 30))),
              "strat": lambda: strat(n, fr, int(rng.integers(1 << 30))), "no floor": lambda: np.zeros(n, bool)}
    print(f"{'layout':<22}" + "".join(f"{k:>10}" for k in floors) + "   (share of HIDDEN needles kept)")
    for ln, lf in layouts.items():
        res = {k: [] for k in floors}
        for _ in range(trials):
            imp = lf()
            visible = imp & (rng.random(n) < detect)
            hidden = imp & ~visible
            if hidden.sum() == 0: continue
            sc = visible * 1.0 + rng.normal(0, 0.5, n)
            for k, ff in floors.items():
                res[k].append(hybrid_floor(sc, rate, ff())[hidden].mean())
        print(f"{ln:<22}" + "".join(f"{np.mean(v):>10.3f}" for v in res.values()))
    print()

def run_tests():
    k = 0
    def check(c, m):
        nonlocal k; assert c, m; k += 1
    ch = sieve_channels(5000, "fixed")
    check(all(exact_channel(n) == ch[n - 1] for n in range(1, 5001)), "exact integer channel equals float channel for small n")
    check(abs(golden(1000, 1/3).mean() - 1/3) < 0.002, "golden keeps 1/3")
    check(np.array_equal(golden(3000, 1/3), sieve_channels(3000, "fixed") == 0), "golden(r=1/3) is the fixed sieve")
    check(strat(300, 1/3).sum() == 100, "stratified keeps exactly one per block")
    check(window_dev(stride(300, 1/3), 1/3, 30) < 1e-9, "stride is perfectly balanced in windows")
    check(len(set(np.diff(np.flatnonzero(ch == 0)).tolist())) <= 3, "three-gap property")
    print(f"all {k} checks passed")

if __name__ == "__main__":
    if "--test" in sys.argv: run_tests()
    else:
        e1_rates(); e2_periodic(); e3_folds(); e4_threeGap_and_prefix(); e5_precision(); e6_hybrid()
