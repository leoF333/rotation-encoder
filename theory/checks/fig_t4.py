"""Figure T4 (gap) : (a) E_OOD vs seuil de troncature, ω_max = 2π/T ; (b) idem à 4π/T (matrices test complètes du calcul
numérique, `results/b1_theory_cache/`, si présentes) ; (c) décomposition de Runge : contribution de chaque
direction généralisée j, (μ_j+2)²/μ_j e_j², en fonction de μ_j ; (d) ridge : E_OOD(κ0) à P = ∞ et P = 1000 vs E_gen (créneau)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
import figstyle as fs
import matplotlib.pyplot as plt
from t4_spectres_reels import load
fs.setup()
OUT = os.path.join(ROOT, "theory", "figures")
TPL = ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]
NAME = dict(gaussian="gaussienne", dipole="dipôle", quarter_circle="quart de cercle", cross="croix", segment="segment", arc="arc")
COL = {t: fs.SERIES[i] for i, t in enumerate(TPL)}      # couleur fixe par template (ordre de la palette)
MRK = dict(gaussian="o", dipole="s", quarter_circle="^", cross="D", segment="v", arc="P")


def trunc_curve(lam, c, S, ct, o2t, nK=40):
    Kmax = int(np.sum(lam[:S.shape[0]] > 1e-14 * lam[0]))
    Ks = np.unique(np.round(np.geomspace(2, Kmax, nK)).astype(int))
    tau = lam[Ks - 1] / lam[0]
    E = []
    for K in Ks:
        W = c[:K] / lam[:K]
        E.append(o2t - 2 * ct[:K] @ W + W @ S[:K, :K] @ W)
    return tau, np.array(E)


fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
# (a) 2π/T
ax = axs[0, 0]
for t in TPL:
    d = load(t, "0.6283")
    tau, E = trunc_curve(d["eigvals"], d["c_proj"], d["test_cov_in_train_basis"], d["test_c_proj"], float(d["test_omega2"]))
    ax.loglog(tau, E, color=COL[t], marker=MRK[t], ms=4, lw=1.6, label=NAME[t] + (" (K ≤ 1000)" if d["K_export"] == 1000 else ""))
vg = (0.6283 / 3) ** 2 / 12
ax.axhline(vg, color=fs.TEXT2, lw=1, ls="--"); ax.text(0.98, vg * 1.25, "Var(ω) dans le gap", color=fs.TEXT2, fontsize=8, ha="right", transform=ax.get_yaxis_transform())
ax.axvline(1e-4, color=fs.GRID, lw=3, zorder=0); ax.axvline(1e-8, color=fs.GRID, lw=3, zorder=0)
ax.text(0.8e-4, 0.97, "τ = 1e-4", color=fs.TEXT2, fontsize=8, va="top", transform=ax.get_xaxis_transform()); ax.text(0.8e-8, 0.97, "τ = 1e-8", color=fs.TEXT2, fontsize=8, va="top", transform=ax.get_xaxis_transform())
ax.invert_xaxis(); ax.set_xlabel("seuil de troncature τ = λ_K/λ_1"); ax.set_ylabel("E_OOD (P = ∞, pinv tronquée)")
ax.set_title("(a) gap, ω_max = 2π/T"); ax.legend(fontsize=8, loc="lower left")
# (b) 4π/T, matrices complètes si disponibles
ax = axs[0, 1]
cache = os.path.join(ROOT, "results", "b1_theory_cache")
for t in TPL:
    f = os.path.join(cache, f"{t}_creneau_n4000_tol1e-14.npz")
    if os.path.exists(f):
        z = np.load(f); lam, c, S, ct, o2t = z["lam"], z["c"], z["test_cov"], z["test_c"], float(z["test_omega2"]); lab = NAME[t]
    else:
        d = load(t, "1.2566"); lam, c, S, ct, o2t = d["eigvals"], d["c_proj"], d["test_cov_in_train_basis"], d["test_c_proj"], float(d["test_omega2"])
        lab = NAME[t] + (" (K ≤ 1000)" if d["K_export"] == 1000 else "")
    tau, E = trunc_curve(lam, c, S, ct, o2t)
    ax.loglog(tau, E, color=COL[t], marker=MRK[t], ms=4, lw=1.6, label=lab)
vg = (1.2566 / 3) ** 2 / 12
ax.axhline(vg, color=fs.TEXT2, lw=1, ls="--"); ax.text(0.98, vg * 1.25, "Var(ω) dans le gap", color=fs.TEXT2, fontsize=8, ha="right", transform=ax.get_yaxis_transform())
ax.invert_xaxis(); ax.set_xlabel("seuil de troncature τ = λ_K/λ_1"); ax.set_ylabel("E_OOD (P = ∞, pinv tronquée)")
ax.set_title("(b) gap, ω_max = 4π/T (pas de dichotomie)"); ax.legend(fontsize=8, loc="lower left")
# (c) décomposition de Runge
ax = axs[1, 0]
for t, K in [("gaussian", None), ("quarter_circle", 1000)]:
    d = load(t, "0.6283")
    r = th.runge_decomposition(d["eigvals"], d["c_proj"], float(d["omega2"]), d["test_cov_in_train_basis"], d["test_c_proj"],
                               float(d["test_omega2"]), K=K)
    mu, e2 = r["mu"], r["e2"]
    m = mu > 1e-6
    contrib = (mu[m] + 2) ** 2 / mu[m] * e2[m]
    ax.loglog(mu[m], contrib, ls="none", marker=MRK[t], ms=5, color=COL[t], mec=fs.SURFACE, mew=0.6,
              label=f"{NAME[t]} (K = {r['mu'].size}) : Σ = {r['shift']:.1e}")
ax.axvline(1, color=fs.TEXT2, lw=1, ls=":"); ax.text(1.3, 0.03, "μ = 1", color=fs.TEXT2, fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xlabel("concentration dans le gap μ_j (énergie test / énergie entraînement)")
ax.set_ylabel("contribution (μ_j+2)²/μ_j · ⟨ε_u, g_j⟩²_train")
ax.set_title("(c) identité de Runge, ω_max = 2π/T"); ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(1e-18, 1e-2)
# (d) ridge
ax = axs[1, 1]
g = 10.0 ** np.arange(-14, -1.9, 0.25)
for t in ["gaussian", "quarter_circle"]:
    d = load(t, "0.6283")
    lam, c, om2 = d["eigvals"], d["c_proj"], float(d["omega2"])
    S, ct, o2t = d["test_cov_in_train_basis"], d["test_c_proj"], float(d["test_omega2"])
    K = S.shape[0]; keep = lam > 1e-15 * lam[0]
    gmin = lam[K - 1] / lam[0] if d["K_export"] == 1000 else 0.0     # en dessous : modes ≥ K_export appris, test tronqué
    gg = g[g >= gmin]
    Einf = [o2t - 2 * ct @ (c[:K] / (lam[:K] + x * lam[0])) + (c[:K] / (lam[:K] + x * lam[0])) @ S @ (c[:K] / (lam[:K] + x * lam[0])) for x in gg]
    E1k = [th.ood_error(lam[keep], c[keep], om2, S, ct, o2t, 1000, x * lam[0] * 1000) for x in gg]
    Eg = [th.gen_error(lam[keep], c[keep], om2, 1000, x * lam[0] * 1000) for x in gg]
    ax.loglog(gg, Einf, color=COL[t], lw=2, label=f"{NAME[t]} : E_OOD, P = ∞")
    ax.loglog(gg, E1k, color=COL[t], lw=1.4, ls="--", label=f"{NAME[t]} : E_OOD, P = 1000")
    ax.loglog(gg, np.maximum(Eg, 1e-18), color=COL[t], lw=1.2, ls=":", label=f"{NAME[t]} : E_gen (créneau), P = 1000")
    i = int(np.argmin(Einf)); ax.plot(gg[i], Einf[i], marker="o", ms=8, color=COL[t], mec=fs.SURFACE, mew=1.5, ls="none")
ax.set_xlabel("ridge effectif κ0 = λ_r/(P λ_1)"); ax.set_ylabel("erreur")
ax.set_title("(d) ridge, 2π/T : optimum du gap ≠ optimum en distribution\n(P = 1000 confondu avec P = ∞ ; ● = optimum du gap)", fontsize=10); ax.legend(fontsize=7.5, loc="lower right")
ax.set_ylim(1e-17, 1e-1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t4_gap_runge.png"), dpi=150)
print("écrit", os.path.join(OUT, "t4_gap_runge.png"))
