"""
sieve_patterns.py - the golden-ratio sieve used as a small PATTERN ENGINE.  numpy only (+ matplotlib for --plot).

    channel(n) = floor(K * frac(n*alpha + offset))     alpha = golden ratio by default, K = number of symbols

Dials:  K (how many symbols), alpha (irrational -> never repeats; rational p/q -> repeats every q), offset (phase).
    pattern1d(n, K, alpha, offset)           1-D strip of symbols
    pattern2d(W, H, K, alpha, beta, mode)    2-D texture; mode "stripes" or "sum"
    hues(n)                                  n well-separated colours (hue values 0..1)

    python sieve_patterns.py          run the experiments
    python sieve_patterns.py --plot   also write sieve_patterns.png
    python sieve_patterns.py --test   self-checks; prints "all N checks passed"
"""
from __future__ import annotations
import sys
from fractions import Fraction
import numpy as np

PHI = (1 + 5 ** 0.5) / 2

def pattern1d(n, K=3, alpha=PHI, offset=0.0, start=1):
    """alpha may be a float, or a Fraction for EXACT rational patterns (floats misround when n*alpha lands on a boundary)."""
    if isinstance(alpha, Fraction):
        off = Fraction(offset).limit_denominator(10 ** 9)
        return np.array([int(((m * alpha + off) % 1) * K) for m in range(start, start + n)], dtype=np.int64) % K
    pos = np.arange(start, start + n, dtype=np.float64)
    return np.floor(K * ((pos * alpha + offset) % 1.0)).astype(np.int64) % K

def pattern2d(W, H, K=3, alpha=PHI, beta=None, mode="sum", offset=0.0):
    beta = PHI ** 2 if beta is None else beta          # frac(phi^2) = frac(phi): use another irrational for variety
    if beta == PHI ** 2: beta = 2 ** 0.5
    x = np.arange(1, W + 1)[None, :]; y = np.arange(1, H + 1)[:, None]
    if mode == "stripes":
        return np.floor(K * ((x * alpha + y * beta + offset) % 1.0)).astype(np.int64) % K
    if mode == "sum":
        return (pattern1d(W, K, alpha, offset)[None, :] + pattern1d(H, K, beta, 0.0)[:, None]) % K
    raise ValueError(mode)

def hues(n, offset=0.0):
    return (np.arange(1, n + 1) * PHI + offset) % 1.0

def min_period(s, max_p):
    """smallest p <= max_p with s[i] == s[i+p] for all i, else None"""
    for p in range(1, max_p + 1):
        if np.array_equal(s[:-p], s[p:]): return p
    return None

# ------------------------------------------------------------------ experiments
def e1_dial():
    print("1  the dial: irrational alpha never repeats, rational alpha = p/q repeats exactly every q steps")
    print(f"  {'alpha':<14}{'first repeat (period)':>24}")
    for nm, a in [("golden", PHI), ("sqrt(2)", 2 ** 0.5), ("3/8", 3 / 8), ("5/13", 5 / 13), ("2/7", 2 / 7), ("1/3 (float)", 1 / 3), ("1/3 (exact)", Fraction(1, 3))]:
        s = pattern1d(3000, 3, a); p = min_period(s, 1000)
        print(f"  {nm:<14}{('none up to 1000' if p is None else str(p)):>24}")
    print("  (a float 1/3 misrounds exactly on channel boundaries and loses its loop; pass Fraction(1, 3) for exact loops)")
    print("  (so one engine spans both families: Euclidean-rhythm style loops and non-repeating patterns)\n")

def e2_no_repeat():
    print("2  immediate repeats (same symbol twice in a row), n=10000")
    rng = np.random.default_rng(0)
    print(f"  {'K':>3}{'golden':>9}{'random':>9}   (share of steps equal to the previous step)")
    for K in range(2, 9):
        g = pattern1d(10000, K); r = rng.integers(0, K, 10000)
        print(f"  {K:>3}{np.mean(g[1:] == g[:-1]):>9.3f}{np.mean(r[1:] == r[:-1]):>9.3f}")
    print()

def e3_hues():
    print("3  colours: closest pair of hues among N, scaled so a perfect even spread = 1.0 (higher is better)")
    rng = np.random.default_rng(1)
    def mind(h):
        h = np.sort(h); d = np.diff(np.r_[h, h[0] + 1]); return d.min()
    print(f"  {'N':>4}{'golden':>9}{'random':>9}")
    for N in [5, 12, 24, 50, 100]:
        print(f"  {N:>4}{mind(hues(N)) * N:>9.2f}{np.mean([mind(rng.random(N)) for _ in range(300)]) * N:>9.2f}")
    print("  (the golden-angle hue trick is well known; this just confirms it holds for any N, and extends one list at a time)\n")

def e4_textures():
    print("4  2-D textures 120x120, K=3: balance and variety of local patches")
    rng = np.random.default_rng(2)
    T = {"stripes": pattern2d(120, 120, mode="stripes"), "sum": pattern2d(120, 120, mode="sum"),
         "periodic (x+y)%3": (np.arange(120)[None, :] + np.arange(120)[:, None]) % 3, "random": rng.integers(0, 3, (120, 120))}
    print(f"  {'texture':<18}{'share of ch 0/1/2':>22}{'distinct 3x3 patches':>22}")
    for nm, t in T.items():
        sh = np.bincount(t.ravel(), minlength=3) / t.size
        patches = {tuple(t[i:i + 3, j:j + 3].ravel()) for i in range(118) for j in range(118)}
        print(f"  {nm:<18}{' '.join(f'{v:.3f}' for v in sh):>22}{len(patches):>22}")
    print("  (random uses nearly every possible patch; periodic uses 3; the sieve sits in between: structured but not repeating)\n")

def e5_cost():
    print("5  determinism and cost: same dials -> same pattern; any position computed alone")
    a = pattern1d(1000, 3); b = pattern1d(1, 3, start=777)[0]
    print(f"  pattern at position 777 computed alone == inside a long strip: {a[776] == b}\n")

# ------------------------------------------------------------------ picture
def plot(path="sieve_patterns.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm3 = ListedColormap(["#1f77b4", "#d62728", "#f2c94c"]); fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    rows = [("golden K=3", pattern1d(80, 3)), ("sqrt2 K=3", pattern1d(80, 3, 2 ** 0.5)), ("3/8 K=3 (loop of 8)", pattern1d(80, 3, 3 / 8)),
            ("golden K=5", pattern1d(80, 5)), ("random K=3", np.random.default_rng(0).integers(0, 3, 80))]
    ax[0, 0].imshow(np.array([r[1] for r in rows]), aspect="auto", cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax[0, 0].set_yticks(range(len(rows))); ax[0, 0].set_yticklabels([r[0] for r in rows]); ax[0, 0].set_title("1. strips: one dial, many patterns")
    ax[0, 1].imshow(pattern2d(100, 100, mode="sum"), cmap=cm3, interpolation="nearest"); ax[0, 1].set_title("4. texture, mode 'sum'")
    ax[0, 2].imshow(pattern2d(100, 100, mode="stripes"), cmap=cm3, interpolation="nearest"); ax[0, 2].set_title("4. texture, mode 'stripes'")
    ax[1, 0].imshow(pattern2d(100, 100, K=5, mode="sum"), cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[1, 0].set_title("texture, K=5, mode 'sum'")
    import colorsys
    for nm, h, y in [("golden", hues(24), 1), ("random", np.random.default_rng(3).random(24), 0)]:
        for i, v in enumerate(h): ax[1, 1].add_patch(plt.Rectangle((i, y), 1, 1, color=colorsys.hsv_to_rgb(v, 0.75, 0.9)))
    ax[1, 1].set_xlim(0, 24); ax[1, 1].set_ylim(0, 2); ax[1, 1].set_yticks([0.5, 1.5]); ax[1, 1].set_yticklabels(["random", "golden"]); ax[1, 1].set_title("3. 24 colours in order")
    g = pattern1d(10000, 3); ax[1, 2].bar(["golden", "random"], [np.mean(g[1:] == g[:-1]), np.mean(np.random.default_rng(4).integers(0, 3, 10000)[1:] == np.random.default_rng(4).integers(0, 3, 10000)[:-1])])
    ax[1, 2].set_title("2. immediate repeats, K=3"); fig.tight_layout(); fig.savefig(path, dpi=100); print("wrote", path)

def run_tests():
    k = 0
    def check(c, m):
        nonlocal k; assert c, m; k += 1
    check(min_period(pattern1d(3000, 3, 3 / 8), 1000) == 8, "alpha=3/8 repeats every 8")
    check(min_period(pattern1d(3000, 3, Fraction(1, 3)), 1000) == 3, "exact alpha=1/3 repeats every 3")
    check(min_period(pattern1d(3000, 3, PHI), 1000) is None, "golden never repeats within 1000")
    check(np.mean(pattern1d(10000, 3)[1:] == pattern1d(10000, 3)[:-1]) == 0, "K=3 golden never repeats a symbol twice in a row")
    check(all(abs(np.mean(pattern1d(10000, K) == j) - 1 / K) < 0.01 for K in (2, 3, 5) for j in range(K)), "symbols are balanced")
    h = np.sort(hues(50)); check(np.diff(np.r_[h, h[0] + 1]).min() * 50 > 0.4, "50 hues are reasonably separated")
    t = pattern2d(60, 60, mode="sum"); check(t.shape == (60, 60) and set(np.unique(t)) == {0, 1, 2}, "2-D texture has the right shape and symbols")
    check(pattern1d(1, 3, start=777)[0] == pattern1d(1000, 3)[776], "position computed alone matches")
    print(f"all {k} checks passed")

if __name__ == "__main__":
    if "--test" in sys.argv: run_tests()
    else:
        e1_dial(); e2_no_repeat(); e3_hues(); e4_textures(); e5_cost()
        if "--plot" in sys.argv: plot()
