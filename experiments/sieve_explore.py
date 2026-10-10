import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pattern_engine import PatternEngine

"""
sieve_explore.py - what can the golden-ratio sieve do, as a stand-alone object?  numpy + scipy (+ matplotlib for --plot).

The sieve:  channel(n) = floor(3 * frac(n * phi)),  n = 1, 2, 3, ...   (phi = golden ratio)
i.e. a 3-way coding of a rotation by the golden ratio.  Equivalent forms: Weyl/Kronecker sequence, "Fibonacci hashing",
the phyllotaxis (sunflower) angle.  Each experiment below tests one use against plain alternatives.

    python sieve_explore.py           run all experiments, print numbers
    python sieve_explore.py --plot    also write sieve_map.png (6-panel picture)
    python sieve_explore.py --test    self-checks; prints "all N checks passed"
"""
from __future__ import annotations
import math, sys
import numpy as np
from scipy.spatial import cKDTree

PHI = (1 + 5 ** 0.5) / 2
def frac_phi(n): return (np.asarray(n, dtype=np.float64) * PHI) % 1.0
def channel(n):  return np.floor(3 * frac_phi(n)).astype(np.int64)

# ---------------------------------------------------------------- A. 2-D map (phyllotaxis)
def spiral_points(N):
    n = np.arange(1, N + 1); r = np.sqrt(n); th = 2 * np.pi * frac_phi(n)       # frac(n*phi) == frac(n/phi)
    return np.c_[r * np.cos(th), r * np.sin(th)]

def nn_stats(P):
    d = cKDTree(P).query(P, k=2)[0][:, 1]
    return d.min() / d.mean(), d.std() / d.mean()          # (higher = more even, lower = more even)

def exp_a(N=1500):
    print("A  2-D map: points at radius sqrt(n), angle 2*pi*frac(n*phi)   (N=%d)" % N)
    rng = np.random.default_rng(0)
    R = np.sqrt(N)
    t = np.sqrt(rng.random(N)) * R; a = rng.uniform(0, 2 * np.pi, N); rand = np.c_[t * np.cos(a), t * np.sin(a)]
    side = int(math.ceil(math.sqrt(N))); G = np.array([(x, y) for x in range(side) for y in range(side)][:N], float)
    print(f"{'layout':<14}{'min/mean NN':>12}{'CV of NN':>10}   (even spread: min/mean near 1, CV near 0)")
    for nm, P in [("golden spiral", spiral_points(N)), ("random", rand), ("square lattice", G)]:
        m, cv = nn_stats(P); print(f"{nm:<14}{m:>12.3f}{cv:>10.3f}")
    print()

# ---------------------------------------------------------------- B. points on a sphere
def fib_sphere(N):
    i = np.arange(N); z = 1 - (2 * i + 1) / N; lon = 2 * np.pi * frac_phi(i + 1); s = np.sqrt(1 - z * z)
    return np.c_[s * np.cos(lon), s * np.sin(lon), z]

def sphere_stats(P, rng):
    T = cKDTree(P); d = T.query(P, k=2)[0][:, 1]; minang = 2 * np.arcsin(d.min() / 2)
    Q = rng.normal(size=(40000, 3)); Q /= np.linalg.norm(Q, axis=1)[:, None]
    cover = 2 * np.arcsin(T.query(Q)[0].max() / 2)
    return np.degrees(minang), np.degrees(cover)

def exp_b(N=1000):
    print("B  points on a sphere: z = 1-(2i+1)/N, longitude = 2*pi*frac(i*phi)   (N=%d)" % N)
    rng = np.random.default_rng(1)
    mg, cg = sphere_stats(fib_sphere(N), rng)
    rs = []
    for _ in range(5):
        P = rng.normal(size=(N, 3)); P /= np.linalg.norm(P, axis=1)[:, None]; rs.append(sphere_stats(P, rng))
    mr, cr = np.mean(rs, axis=0)
    print(f"{'layout':<10}{'closest pair (deg)':>20}{'largest hole (deg)':>20}   (want: closest pair big, hole small)")
    print(f"{'golden':<10}{mg:>20.2f}{cg:>20.2f}\n{'random':<10}{mr:>20.2f}{cr:>20.2f}\n")

# ---------------------------------------------------------------- C. integration
def exp_c():
    print("C  numerical integration with x_n = frac(n*phi)  (rms error over 100 trials for random)")
    rng = np.random.default_rng(2)
    fs = {"exp(x)": (np.exp, math.e - 1), "step x<0.3": (lambda x: (x < 0.3) * 1.0, 0.3)}
    Ns = [30, 100, 300, 1000, 3000, 10000, 30000]
    for nm, (f, true) in fs.items():
        print(f"  f = {nm}")
        print(f"  {'N':>7}{'golden err':>13}{'random err':>13}")
        eg, er = [], []
        for N in Ns:
            g = abs(f(frac_phi(np.arange(1, N + 1))).mean() - true)
            r = math.sqrt(np.mean([(f(rng.random(N)).mean() - true) ** 2 for _ in range(100)]))
            eg.append(g); er.append(r); print(f"  {N:>7}{g:>13.2e}{r:>13.2e}")
        if nm == "exp(x)":
            sg = np.polyfit(np.log(Ns), np.log(eg), 1)[0]; sr = np.polyfit(np.log(Ns), np.log(er), 1)[0]
            print(f"  log-log slope: golden {sg:.2f}, random {sr:.2f}  (random is about -0.5; golden errors are not monotone but stay far below)")
        else: print("  (golden error is 0 or about 1/N at every N shown; no slope fitted because exact zeros break a log fit)")
    print()

# ---------------------------------------------------------------- D. hashing / bucket balance
def exp_d(N=3000):
    print("D  3-way bucket assignment of structured keys key = c*i (i=0..%d), worst imbalance over c=1..60" % (N - 1))
    i = np.arange(N); worst = {"mod 3": (0, 0), "golden": (0, 0)}
    for c in range(1, 61):
        keys = c * i
        for nm, b in [("mod 3", keys % 3), ("golden", channel(keys))]:
            cnt = np.bincount(b, minlength=3); imb = (cnt.max() - cnt.min()) / N
            if imb > worst[nm][0]: worst[nm] = (imb, c)
    for nm, (v, c) in worst.items(): print(f"  {nm:<8} worst imbalance {v:.4f} at c={c}")
    print("  (imbalance = (largest bucket - smallest bucket)/N; mod 3 puts everything in one bucket whenever c is a multiple of 3)\n")

# ---------------------------------------------------------------- E. rhythm / scheduling
def exp_e(steps=34):
    print("E  as a 3-voice pattern (x = voice plays on that step); voices never collide and are evenly spaced")
    ch = channel(np.arange(1, steps + 1))
    for v in range(3):
        row = "".join("x" if c == v else "." for c in ch); g = np.diff(np.flatnonzero(ch == v))
        print(f"  voice {v}: {row}   gaps used {sorted(set(g.tolist()))}")
    print()

# ---------------------------------------------------------------- F. spectrum (long-range order)
def spectrum(x):
    X = np.abs(np.fft.rfft(x - x.mean())) / len(x); return X

def exp_f(N=4096):
    print("F  spectrum of the channel-0 pattern (N=%d): sharp lines at positions set by the golden ratio" % N)
    x = (channel(np.arange(1, N + 1)) == 0).astype(float); X = spectrum(x)
    st = (np.arange(N) % 3 == 0).astype(float); Xs = spectrum(st)
    rng = np.random.default_rng(3); Xr = spectrum((rng.random(N) < 1 / 3).astype(float))
    for nm, S in [("golden", X), ("stride-3", Xs), ("random", Xr)]:
        print(f"  {nm:<9} tallest line / median line = {S.max() / np.median(S):>8.1f}")
    print(f"  {'k':>3}{'predicted freq':>16}{'predicted amp':>15}{'measured freq':>15}{'amp at that freq':>18}")
    freqs = np.fft.rfftfreq(N)
    nn = np.arange(1, N + 1)
    for k in [1, 2, 4, 5, 3]:
        f = frac_phi(k); f = min(f, 1 - f); amp = abs(math.sin(math.pi * k / 3)) / (math.pi * k)
        j = int(round(f * N)); lo, hi = max(j - 3, 0), min(j + 4, len(X)); jm = lo + int(np.argmax(X[lo:hi]))
        direct = abs(np.mean((x - x.mean()) * np.exp(-2j * np.pi * f * nn)))
        print(f"  {k:>3}{f:>16.4f}{amp:>15.4f}{freqs[jm]:>15.4f}{direct:>14.4f}")
    print("  (k=3 has predicted amplitude 0 because the keep share is exactly 1/3)\n")

# ---------------------------------------------------------------- G. dithering
def vdc(n, base=2):
    out = np.zeros(len(n)); f = 1.0 / base; m = n.copy()
    while m.any():
        out += f * (m % base); m //= base; f /= base
    return out

def exp_g(N=4096):
    print("G  1-bit dithering of a smooth ramp: rms error of local averages (lower is better)")
    n = np.arange(1, N + 1); v = n / (N + 1); rng = np.random.default_rng(4)
    th = {"fixed 0.5": np.full(N, 0.5), "random": rng.random(N), "golden": frac_phi(n), "van der Corput": vdc(n.copy())}
    print(f"  {'threshold':<16}" + "".join(f"{'W=%d' % W:>10}" for W in (8, 32, 128)))
    for nm, t in th.items():
        out = (v > t).astype(float); row = []
        for W in (8, 32, 128):
            k = np.ones(W) / W
            row.append(math.sqrt(np.mean((np.convolve(out, k, 'valid') - np.convolve(v, k, 'valid')) ** 2)))
        print(f"  {nm:<16}" + "".join(f"{e:>10.4f}" for e in row))
    print()

# ---------------------------------------------------------------- H. pattern complexity
def exp_h(N=20000):
    print("H  pattern complexity: number of distinct length-L words in the channel sequence")
    s = channel(np.arange(1, N + 1)); st = np.arange(N) % 3
    print(f"  {'L':>3}{'golden':>8}{'stride-3':>10}{'max possible':>14}")
    for L in range(1, 11):
        w = lambda a: len({tuple(a[i:i + L]) for i in range(len(a) - L)})
        print(f"  {L:>3}{w(s):>8}{w(st):>10}{3 ** L:>14}")
    print("  (never repeats, yet has very few distinct patterns: the signature of a quasi-periodic sequence)\n")

# ---------------------------------------------------------------- picture
def plot(path="sieve_map.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = np.array(["#1f77b4", "#d62728", "#7f7f7f"]); fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    P = spiral_points(900); c = channel(np.arange(1, 901))
    ax[0, 0].scatter(P[:, 0], P[:, 1], c=cols[c], s=14); ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("A. spiral map, coloured by channel")
    n = np.arange(1, 91); ch = channel(n)
    for v in range(3): ax[0, 1].scatter(n[ch == v], np.full((ch == v).sum(), v), c=cols[v], s=30)
    ax[0, 1].set_yticks([0, 1, 2]); ax[0, 1].set_title("E. the 3 channels, tokens 1-90 (no collisions, even gaps)")
    N = 4096; x = (channel(np.arange(1, N + 1)) == 0).astype(float); X = spectrum(x); fr = np.fft.rfftfreq(N)
    ax[0, 2].plot(fr, X, lw=0.7, label="golden"); ax[0, 2].plot(fr, spectrum((np.arange(N) % 3 == 0).astype(float)), lw=0.7, label="stride-3", alpha=0.7)
    ax[0, 2].set_title("F. spectrum: sharp lines, never repeating"); ax[0, 2].legend()
    Ns = np.unique(np.logspace(1, 4.5, 30).astype(int)); rng = np.random.default_rng(5)
    eg = [abs(np.exp(frac_phi(np.arange(1, k + 1))).mean() - (math.e - 1)) for k in Ns]
    er = [math.sqrt(np.mean([(np.exp(rng.random(k)).mean() - (math.e - 1)) ** 2 for _ in range(30)])) for k in Ns]
    ax[1, 0].loglog(Ns, eg, label="golden"); ax[1, 0].loglog(Ns, er, label="random"); ax[1, 0].legend(); ax[1, 0].set_title("C. integration error vs N")
    S = fib_sphere(700); up = S[:, 2] >= 0
    ax[1, 1].scatter(S[up, 0], S[up, 1], s=12, c=S[up, 2], cmap="viridis"); ax[1, 1].set_aspect("equal"); ax[1, 1].set_title("B. sphere points, seen from above")
    cs = np.arange(1, 61); i = np.arange(3000)
    im = lambda f: [(np.bincount(f(c * i), minlength=3).max() - np.bincount(f(c * i), minlength=3).min()) / 3000 for c in cs]
    ax[1, 2].plot(cs, im(lambda k: k % 3), label="mod 3"); ax[1, 2].plot(cs, im(channel), label="golden"); ax[1, 2].legend()
    ax[1, 2].set_title("D. bucket imbalance for keys c*i"); ax[1, 2].set_xlabel("c")
    fig.tight_layout(); fig.savefig(path, dpi=110); print("wrote", path)

def run_tests():
    k = 0
    def check(c, m):
        nonlocal k; assert c, m; k += 1
    m, cv = nn_stats(spiral_points(1500)); rng = np.random.default_rng(0)
    t = np.sqrt(rng.random(1500)) * math.sqrt(1500); a = rng.uniform(0, 6.2832, 1500)
    m2, cv2 = nn_stats(np.c_[t * np.cos(a), t * np.sin(a)])
    check(m > 3 * m2 and cv < cv2, "spiral is more even than random points")
    check(sphere_stats(fib_sphere(1000), np.random.default_rng(1))[0] > 3 * 1.0, "sphere points keep a sensible minimum separation")
    check(abs(np.exp(frac_phi(np.arange(1, 10001))).mean() - (math.e - 1)) < 1e-3, "integration with golden points is accurate at N=10000")
    i = np.arange(3000)
    check(all((np.ptp(np.bincount(channel(c * i), minlength=3)) / 3000) < 0.02 for c in range(1, 61)), "golden buckets within 2% for every multiplier 1..60 (measured worst: 1.3%)")
    check(np.ptp(np.bincount((3 * i) % 3, minlength=3)) == 3000, "mod 3 fails on multiples of 3 (control)")
    N = 4096; x = (channel(np.arange(1, N + 1)) == 0).astype(float); nn = np.arange(1, N + 1)
    amp = lambda kk: abs(np.mean((x - x.mean()) * np.exp(-2j * np.pi * frac_phi(kk) * nn)))
    check(abs(amp(1) - math.sin(math.pi / 3) / math.pi) < 0.01, "line at k=1 has the predicted height")
    check(abs(amp(2) - math.sin(2 * math.pi / 3) / (2 * math.pi)) < 0.01, "line at k=2 has the predicted height")
    check(amp(3) < 0.005, "line at k=3 is absent")
    check(len({tuple(channel(np.arange(1, 5001))[i:i + 6]) for i in range(4900)}) == 18, "complexity is 3L (L=6 -> 18 words)")
    print(f"all {k} checks passed")

if __name__ == "__main__":
    if "--test" in sys.argv: run_tests()
    else:
        exp_a(); exp_b(); exp_c(); exp_d(); exp_e(); exp_f(); exp_g(); exp_h()
        if "--plot" in sys.argv: plot()
