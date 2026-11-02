"""T5 — Recalcul de theory.ood_error (matrices test COMPLÈTES du cache numérique, 4π/T) contre les simulations B1
(`results/b1_aggregate.csv`, médianes sur 5 tirages). Lecture seule des fichiers de la partie numérique."""
import sys, os, csv, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
warnings.simplefilter("ignore", RuntimeWarning)
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "b1_aggregate.csv")))
        if r["dist"] == "creneau" and r["reg"].startswith("ridge")]
cache, res = {}, []
for r in rows:
    t = r["template"]
    if t not in cache:
        z = np.load(os.path.join(ROOT, "results", "b1_theory_cache", f"{t}_creneau_n4000_tol1e-14.npz"))
        cache[t] = (z["lam"], z["c"], float(z["omega2"]), z["test_cov"], z["test_c"], float(z["test_omega2"]))
    lam, c, o2, S, ct, o2t = cache[t]
    P = int(float(r["P"])); rr = float(r["reg"][5:])
    E = th.ood_error(lam, c, o2, S, ct, o2t, P, rr * P * lam[0])
    res.append((t, r["kind"], rr, P, float(r["ood_median"]) / E - 1))
for kind in ["real", "surrogate"]:
    for Pmin, Pmax in [(300, 8000), (70, 90), (0, 60)]:
        v = np.array([x[4] for x in res if x[1] == kind and Pmin <= x[3] <= Pmax])
        print(f"{kind:>9} P∈[{Pmin},{Pmax}] n={v.size} médiane |écart|={np.median(np.abs(v)):.3f} max |écart|={np.abs(v).max():.2f}")
print("écarts > 5 % (réel, P ≥ 300) :")
for x in res:
    if x[1] == "real" and x[3] >= 300 and abs(x[4]) > 0.05:
        s = [y for y in res if y[0] == x[0] and y[1] == "surrogate" and y[2] == x[2] and y[3] == x[3]]
        print(f"   {x[0]:>15} r={x[2]:.0e} P={x[3]:5d} réel {x[4]:+.2f}  substitut {s[0][4] if s else float('nan'):+.2f}")
