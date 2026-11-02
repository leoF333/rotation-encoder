"""B3 — analyse du balayage d'architectures (results/b3/*.json, cf. scripts/b3_architectures.py).

Sorties : results/b3_summary.md (tableau moyenne ± écart-type sur les seeds, front de Pareto),
figures/b3_pareto.png, figures/b3_latents.png.
Usage : python scripts/b3_analysis.py [--template arc]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aerot.paths import FIG, RES
from aerot.plotting import setup, save, plt, PALETTE, INK2, GAP_SHADE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_architectures import CONFIGS, GA, GB  # noqa: E402


def latent_quality(r):
    """Qualité du latent sur tout l'intervalle : R² monotone (k = 1) ou R² monotone de l'angle (k = 2)."""
    if r["k"] == 1:
        return r.get("R2_monotone_all", np.nan)
    return r.get("R2_monotone_angle_all", np.nan)


def pareto(points):
    """Indices non dominés (minimisation de toutes les coordonnées)."""
    P = np.asarray(points, float)
    keep = []
    for i, p in enumerate(P):
        if not np.any(np.all(P <= p, axis=1) & np.any(P < p, axis=1)):
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="arc")
    a = ap.parse_args()
    setup()
    rows = []
    for f in glob.glob(str(RES / "b3" / f"{a.template}_*.json")):
        r = json.load(open(f))
        r["latq"] = latent_quality(r)
        r["latq_gap"] = r.get("R2_monotone_gap", np.nan) if r["k"] == 1 else r.get("R2_monotone_angle_gap", np.nan)
        # linéarité de z(ω) (k = 1) : si z ≈ a ω + b, l'encodeur linéaire réalise la régression de ω étudiée par les répliques
        r["R2_lin"] = np.nan
        npz = Path(f).with_suffix(".npz")
        if r["k"] == 1 and npz.exists():
            z = np.load(npz)
            zz, om = z["z_all"].reshape(-1), z["om_all"]
            if np.std(zz) > 0:
                r["R2_lin"] = float(np.corrcoef(zz, om)[0, 1] ** 2)
        rows.append(r)
    df = pd.DataFrame(rows)
    order = [c for c in CONFIGS if c in set(df.config)]
    agg = []
    for c in order:
        g = df[df.config == c]
        e = dict(config=c, k=int(g.k.iloc[0]), P=int(g.P.iloc[0]), n_seeds=len(g), n_params=int(g.n_params.iloc[0]),
                 collapsed=int(g.collapsed.sum()))
        for q in ["mse_ind", "mse_gap", "latq", "R2_knn_all", "latq_gap", "R2_lin", "train_time_s"]:
            if q in g:
                v = g[q].astype(float)
                e[q] = v.median()
                e[q + "_mean"] = v.mean()
                e[q + "_std"] = v.std(ddof=0)
                e[q + "_min"] = v.min()
        agg.append(e)
    A = pd.DataFrame(agg)
    # front de Pareto (médianes) : (MSE distribution, MSE gap, 1 − qualité du latent au pire seed)
    obj = np.c_[np.log10(A.mse_ind), np.log10(A.mse_gap), np.log10(1 - A.latq_min.clip(upper=1 - 1e-9))]
    pf = pareto(obj)
    A["pareto"] = False
    A.loc[pf, "pareto"] = True
    lines = [f"# B3 — balayage d'architectures (template {a.template})", "",
             "Entraînement sur le créneau de [0.05, 2π/T] (gap = tiers central), P vidéos, 6400 pas d'Adam, 3 seeds. "
             "Valeurs : médiane sur les seeds (± écart-type) ; « latent » = R² de la meilleure fonction monotone ω = f(z) "
             "sur tout l'intervalle (angle polaire si k = 2), « pire seed » = minimum ; « effondrés » = latent constant. "
             "Pareto : non dominé sur (MSE distribution, MSE gap, 1 − latent au pire seed).", "",
             "| config | k | P | paramètres | MSE distrib. | MSE gap | latent (méd.) | latent (pire seed) | R² 5-NN | "
             "R² monotone dans le gap | R² linéaire z(ω) | temps (s) | effondrés | Pareto |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, e in A.iterrows():
        lines.append(f"| {e.config} | {e.k} | {e.P} | {e.n_params:.2e} | {e.mse_ind:.2e} ± {e.mse_ind_std:.1e} | "
                     f"{e.mse_gap:.2e} ± {e.mse_gap_std:.1e} | {e.latq:.5f} | {e.latq_min:.4f} | {e.R2_knn_all:.4f} | "
                     f"{e.latq_gap:.4f} | {e.R2_lin:.3f} | {e.train_time_s:.0f} | {e.collapsed}/{e.n_seeds} | {'**oui**' if e.pareto else ''} |")
    (RES / f"b3_summary_{a.template}.md").write_text("\n".join(lines) + "\n")
    A.to_csv(RES / f"b3_summary_{a.template}.csv", index=False)
    print("\n".join(lines))

    # ---- figure Pareto
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    fam_col = {"lin": PALETTE[0], "mlp": PALETTE[1], "deep": PALETTE[2]}
    for _, e in A.iterrows():
        fam = e.config.split("-")[0]
        col = fam_col.get(fam, INK2)
        mk = "s" if e.k == 2 else ("^" if e.P < 4000 else "o")
        sz = 20 + 30 * np.log10(e.n_params / 1e4)
        for ax, (x, y) in zip(axs, [(e.mse_ind, e.mse_gap), (e.mse_ind, 1 - e.latq_min)]):
            ax.scatter(x, y, s=sz, color=col, marker=mk, edgecolor="#0b0b0b" if e.pareto else "none", lw=1.2, alpha=0.9)
            ax.annotate(e.config, (x, y), fontsize=5.5, xytext=(3, 2), textcoords="offset points", color=INK2)
    axs[0].set_xlabel("MSE de reconstruction en distribution (médiane)")
    axs[0].set_ylabel("MSE de reconstruction dans le gap")
    axs[1].set_xlabel("MSE de reconstruction en distribution (médiane)")
    axs[1].set_ylabel("1 − R² monotone du latent (pire seed)")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
    for fam, col in fam_col.items():
        axs[0].scatter([], [], color=col, label={"lin": "encodeur linéaire", "mlp": "encodeur 1 couche cachée",
                                                  "deep": "encodeur 2 couches cachées"}[fam])
    axs[0].scatter([], [], marker="s", color=INK2, label="latent k = 2")
    axs[0].scatter([], [], marker="^", color=INK2, label="P < 4000")
    axs[0].scatter([], [], marker="o", color="white", edgecolor="#0b0b0b", label="front de Pareto")
    axs[0].legend(fontsize=6)
    fig.suptitle(f"B3 — reconstruction vs qualité du latent (template {a.template} ; taille ∝ log nombre de paramètres)",
                 fontsize=9)
    save(fig, FIG / f"b3_pareto_{a.template}.png", dict(template=a.template, pareto=list(A.config[A.pareto])))

    # ---- figure latents (seed 0)
    show = [c for c in ["lin-shallow", "lin-deep", "mlp-deep", "deep-deep", "lin-deep-k2", "mlp-deep-P500"] if c in order]
    fig, axs = plt.subplots(1, len(show), figsize=(2.9 * len(show), 2.8))
    for ax, c in zip(np.atleast_1d(axs), show):
        f = RES / "b3" / f"{a.template}_{c}_s0.npz"
        if not f.exists():
            continue
        z = np.load(f)
        om, zz = z["om_all"], z["z_all"]
        ax.axvspan(GA, GB, color=GAP_SHADE, alpha=0.10, lw=0)
        if zz.ndim == 2 and zz.shape[1] == 2:
            ang = np.unwrap(np.arctan2(zz[:, 1] - zz[:, 1].mean(), zz[:, 0] - zz[:, 0].mean()))
            ax.scatter(om, ang, s=2, color=PALETTE[0])
            ax.set_ylabel("angle de (z₁, z₂)")
        else:
            ax.scatter(om, zz.reshape(-1), s=2, color=PALETTE[0])
            ax.set_ylabel("z")
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("ω")
    save(fig, FIG / f"b3_latents_{a.template}.png", dict(template=a.template, seed=0, configs=show))


if __name__ == "__main__":
    main()
