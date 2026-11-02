"""Figures T3 : (a) double descente théorie vs simulation gaussienne ; (b) lois d'échelle (spectre puissance vs exp)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
import figstyle as fs
import matplotlib.pyplot as plt
fs.setup()
rng = np.random.default_rng(11)
OUT = os.path.join(ROOT, "theory", "figures")

# (a) double descente, N = 200, λ_k = k^-1.2, t_k = 0.5 k^-1.5, σ² = 0.02
N = 200
k = np.arange(1, N + 1)
lam = k ** -1.2; t = 0.5 * k ** -1.5; c = np.sqrt(t * lam) * rng.choice([-1, 1], N); s2 = 0.02
om2 = t.sum() + s2
Pg = np.unique(np.r_[np.linspace(10, 600, 150), N - 1, N + 1]).astype(float)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, lr in enumerate([0.0, 1e-2, 1e-1, 1.0]):
    E = np.array([th.gen_error(lam, c, om2, P, lr) if P != N else np.nan for P in Pg])
    ax[0].plot(Pg, E, color=fs.SERIES[i], label=f"$\\lambda_r$ = {lr:g}")
    # simulation gaussienne
    Ps = [40, 120, 180, 220, 300, 500]
    em = []
    for P in Ps:
        v = []
        for _ in range(30):
            X = rng.standard_normal((P, N)) * np.sqrt(lam)
            y = X @ (c / lam) + np.sqrt(s2) * rng.standard_normal(P)
            W = (np.linalg.lstsq(X, y, rcond=None)[0] if lr == 0 else
                 np.linalg.solve(X.T @ X + lr * np.eye(N), X.T @ y))
            v.append(s2 + np.sum(lam * (W - c / lam) ** 2))
        em.append(np.mean(v))
    ax[0].plot(Ps, em, "o", color=fs.SERIES[i], markeredgecolor=fs.SURFACE, markeredgewidth=1.0)
ax[0].axvline(N, color=fs.TEXT2, lw=0.8, ls=":")
ax[0].text(N * 1.1, 1.6, "P = N (pic)", color=fs.TEXT2)
ax[0].set_yscale("log"); ax[0].set_ylim(0.02, 3)
ax[0].set_xlabel("P (nombre d'échantillons)"); ax[0].set_ylabel("$E_{\\rm gen}$")
ax[0].set_title("(a) Double descente : théorie (traits) vs simulation (points)", loc="left")
ax[0].legend(loc="upper right")

# (b) lois d'échelle au ridge optimal et ridgeless : spectre puissance vs exponentiel, même cible t_k = k^-3
kk = np.arange(1, 6001).astype(float)
t = kk ** -3.0; s2 = 1e-5
Ps = np.unique(np.logspace(1, 3.3, 20).astype(int)).astype(float)
for i, (name, lamk) in enumerate([("puissance $k^{-2}$", kk ** -2.0), ("exponentiel $e^{-0.1k}$", np.exp(-0.1 * kk))]):
    cc = np.sqrt(t * lamk)
    Er = np.array([th.gen_error(lamk, cc, t.sum() + s2, P, 0.0) for P in Ps])
    Eo = np.array([th.optimal_ridge(lamk, cc, t.sum() + s2, P)[1] for P in Ps])
    ax[1].plot(Ps, Er - s2, color=fs.SERIES[i], ls="--", label=f"{name}, ridgeless")
    ax[1].plot(Ps, Eo - s2, color=fs.SERIES[i], label=f"{name}, ridge optimal")
ax[1].plot(Ps, 3e-5 * (Ps / 10) ** (-2 / 3), color=fs.TEXT2, lw=1, ls=":")
ax[1].text(Ps[-6], 3e-5 * (Ps[-6] / 10) ** (-2 / 3) * 1.6, "$P^{-(b-1)/b}$", color=fs.TEXT2)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("P"); ax[1].set_ylabel("$E_{\\rm gen}-\\sigma^2_{\\rm res}$")
ax[1].set_title("(b) Même cible ($t_k=k^{-3}$), $\\sigma^2_{\\rm res}=10^{-5}$", loc="left")
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t3_double_descente_lois_echelle.png"), dpi=150)
print("ok")
