"""T5 — Écart à l'équivalence gaussienne sur les vraies données de rotation (B1 de la partie numérique) :
test du critère « trous d'échantillonnage » : l'équivalence gaussienne échoue quand le plus grand trou entre les P valeurs
de ω tirées, ≈ ω_max ln P / P, n'est pas petit devant l'échelle de résolution des modes appris ≈ ω_max / df(κ),
soit x = df(κ) ln P / P ≳ 1 (mini-gaps : mécanisme de T4 à petite échelle).
Lit results/b1_aggregate.csv (lecture seule) et les spectres data/spectra (ω_max = 4π/T, uniforme)."""
import sys, os, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "b1_aggregate.csv"))))
spec = {}
out = []
for r in rows:
    if r["dist"] != "uniform" or not r["reg"].startswith("ridge"):
        continue
    t = r["template"]
    if t not in spec:
        z = np.load(os.path.join(ROOT, "data", "spectra", f"{t}_uniform_d28_T10_wmax1.2566.npz"))
        lam = z["eigvals"]; k = lam > 1e-14 * lam[0]
        spec[t] = (lam[k], z["c_proj"][k], float(z["omega2"]))
    lam, c, o2 = spec[t]
    P = int(float(r["P"])); rr = float(r["reg"][5:])
    op = th.order_parameters(lam, c, o2, P, rr * P * lam[0])
    kind_rows = [q for q in rows if q["template"] == t and q["dist"] == "uniform" and q["reg"] == r["reg"] and q["P"] == r["P"]]
    real = [q for q in kind_rows if q["kind"] == "real"]
    sur = [q for q in kind_rows if q["kind"] == "surrogate"]
    if r["kind"] != "real" or not sur:
        continue
    x = op["df"] * np.log(P) / P
    out.append((t, rr, P, op["df"], x, float(real[0]["gen_median"]) / op["Egen"], float(sur[0]["gen_median"]) / op["Egen"]))
out.sort(key=lambda o: o[4])
print(f"{'template':>15} {'r':>6} {'P':>5} {'df':>6} {'x=df lnP/P':>10} | réel/théorie  substitut/théorie")
for o in out:
    print(f"{o[0]:>15} {o[1]:6.0e} {o[2]:5d} {o[3]:6.1f} {o[4]:10.2f} | {o[5]:9.2f}  {o[6]:9.2f}")
xs = np.array([o[4] for o in out]); yr = np.array([o[5] for o in out]); ys = np.array([o[6] for o in out])
for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 100)]:
    m = (xs >= lo) & (xs < hi)
    if m.any():
        print(f"x ∈ [{lo},{hi}) : n={m.sum():2d}  médiane réel/théorie = {np.median(yr[m]):.2f} (max {yr[m].max():.1f}) ; substitut {np.median(ys[m]):.2f}")

# ---- figure : rapport réel/théorie (médiane) vs x = df ln P / P, un panneau par template ----
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figstyle as fs
    import matplotlib.pyplot as plt
    fs.setup()
    SEQ = {1e-2: "#9ec5f4", 1e-4: "#5598e7", 1e-6: "#256abf", 1e-8: "#0d366b"}   # rampe bleue : ridge décroissant
    NAME = dict(gaussian="gaussienne", quarter_circle="quart de cercle", cross="croix", segment="segment")
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, t in zip(axs, ["gaussian", "quarter_circle", "cross", "segment"]):
        sel = [o for o in out if o[0] == t and o[2] >= 10]
        ax.scatter([o[4] for o in sel], [o[6] for o in sel], s=22, facecolors="none", edgecolors=fs.TEXT2, lw=0.8,
                   label="substitut gaussien")
        for rr, col in SEQ.items():
            s2 = [o for o in sel if o[1] == rr]
            ax.scatter([o[4] for o in s2], [o[5] for o in s2], s=30, color=col, edgecolors=fs.SURFACE, lw=0.6,
                       label=f"rotation, r = {rr:.0e}")
        ax.axvline(1, color=fs.TEXT2, lw=1, ls=":"); ax.axhline(1, color=fs.TEXT2, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(NAME[t])
        ax.set_xlabel("x = df(κ) ln P / P")
    axs[0].set_ylabel("E_gen médian / théorie (loi uniforme, 4π/T)")
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Équivalence gaussienne sur la variété de rotation : valable tant que le plus grand trou d'échantillonnage "
                 "(≈ ω_max ln P/P) reste petit devant ω_max/df (x < 1)", fontsize=10)
    fig.tight_layout()
    f = os.path.join(ROOT, "theory", "figures", "t5_equivalence_gaussienne.png")
    fig.savefig(f, dpi=150); print("écrit", f)
