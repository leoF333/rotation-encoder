import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import vlib
wmax = 2 * np.pi / vlib.T
om = np.linspace(0, wmax, 1201); gap = (om > wmax / 3) & (om < 2 * wmax / 3)
# robustesse de la gaussienne
for lab, n, ss in [("base", 4000, 4), ("n=8000", 8000, 4), ("ss=8", 4000, 8)]:
    pop = vlib.population_svd(vlib.gaussian, "creneau", wmax, n=n, ss=ss)
    W, K = vlib.truncated_W(pop, 1e-8)
    E = (om - vlib.render(vlib.gaussian, om, ss=ss) @ W) ** 2
    print(f"gaussienne {lab:7s} 1e-8 : K={K} pic gap={E[gap].max():.2e}", flush=True)
# figure 1 : E(ω) recalculé, 2π/T, coupure 1e-8 (convention qui reproduit le mémoire)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
labels = dict(gaussian="gaussienne", segment="segment", quarter_circle="quart de cercle", cross="croix")
memo = dict(gaussian=9.3e-6, segment=1.5e-3, quarter_circle=1.2e-3, cross=1.3e-2)
for a, nm in zip(ax, labels):
    E = np.load(f"results/v2_E_{nm}_2piT.npy") if False else None
    pop = vlib.population_svd(vlib.TEMPLATES[nm], "creneau", wmax, n=4000)
    W, K = vlib.truncated_W(pop, 1e-8)
    E = (om - vlib.render(vlib.TEMPLATES[nm], om) @ W) ** 2
    a.axvspan(wmax / 3, 2 * wmax / 3, color="#e8a0a0", alpha=0.35, lw=0)
    a.semilogy(om, np.maximum(E, 1e-12), color="#8b1a1a", lw=1.4)
    a.axhline(memo[nm], color="0.3", ls="--", lw=1)
    a.set_title(f"{labels[nm]} (K = {K})", fontsize=10); a.set_xlabel("ω (rad/image)")
    a.set_ylim(1e-10, 1e-1); a.grid(alpha=0.3)
ax[0].set_ylabel("E(ω) = (ω − W·φ(ω))²")
fig.suptitle("Recalcul indépendant : erreur ponctuelle, ω_max = 2π/T, pinv tronquée λ/λ₁ > 1e-8 (tirets : pic du mémoire)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/v2_gap_recalcul.png", dpi=130)
# figure 2 : V3 théorie vs simulation
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for nm, mk in [("gaussian", "o"), ("cross", "s")]:
    rows = json.load(open(f"results/v3_ood_{nm}.json"))
    for key, a in [("Eood", ax[0]), ("Egen", ax[1])]:
        th = np.array([r[f"{key}_th"] for r in rows]); sm = np.array([r[f"{key}_sim"] for r in rows])
        x = np.array([r["x"] for r in rows])
        sc = a.scatter(th, sm, c=np.log10(x), cmap="viridis", marker=mk, vmin=-1.7, vmax=0.7, label=labels[nm], edgecolor="k", lw=0.4)
for a, t in zip(ax, ["E_OOD (gap)", "E_gen (créneau, en distribution)"]):
    lo, hi = a.get_xlim()[0], a.get_xlim()[1]
    a.set_xscale("log"); a.set_yscale("log")
    l = np.array([1e-9, 1e-2]); a.plot(l, l, "k-", lw=0.8); a.plot(l, 2 * l, "k:", lw=0.6); a.plot(l, l / 2, "k:", lw=0.6)
    a.set_xlabel("théorie des répliques (ré-implémentée)"); a.set_ylabel("simulation ridge, vraies vidéos"); a.set_title(t, fontsize=10); a.grid(alpha=0.3)
ax[0].set_xlim(1e-5, 1e-2); ax[0].set_ylim(1e-5, 1e-2); ax[1].set_xlim(1e-9, 1e-3); ax[1].set_ylim(1e-9, 1e-3)
ax[0].legend(fontsize=8); cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("log10 x,  x = df(κ) ln P / P")
fig.savefig("figures/v3_theorie_vs_simulation.png", dpi=130, bbox_inches="tight")
print("figures OK")
