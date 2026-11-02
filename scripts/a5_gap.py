"""A5 — expérience de gap (Sec. 2.2, Fig. 6-7 du mémoire).

Encodeur linéaire W entraîné par régression sur la distribution créneau (population : quadrature
de Gauss-Legendre ; ou P échantillons i.i.d.), puis E(omega) = (omega - W.phi(omega))^2 sur [0, omega_max].
Régresseurs : pseudo-inverse tronquée (seuil relatif sur lambda_k/lambda_1 : 1e-4 comme le mémoire,
1e-8, 1e-12) et ridge (lam_r = r * lambda_1, r = 1e-3, 1e-6, 1e-9).

Sorties : results/a5_gap_<template>_wmax<..>.npz, results/a5_gap_summary.md,
figures/a5_gap_memoir_<wmax>.png (format mémoire), figures/a5_gap_regressors_<wmax>.png.
Usage : python scripts/a5_gap.py [--wmax 4] [--n 4000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from aerot import render, spectra, templates
from aerot.paths import FIG, RES
from aerot.plotting import setup, save, plt, TEMPLATE_COLOR, PALETTE, GAP_SHADE

TRUNC = [1e-4, 1e-8, 1e-12]
RIDGE = [1e-3, 1e-6, 1e-9]


def gap_curves(name, wmax, n=4000, d=28, T=10, method=render.DEFAULT_METHOD, ss=render.DEFAULT_SS, n_grid=1201, P_list=(2000,),
               seeds=(0, 1, 2), params=None):
    om, w = render.quadrature("creneau", wmax, n)
    Phi = render.make_videos(om, name, d=d, T=T, method=method, ss=ss, params=params).numpy()
    S = spectra.GramSpectrum(Phi, w, om)
    og = np.linspace(0, wmax, n_grid)
    Pg = render.make_videos(og, name, d=d, T=T, method=method, ss=ss, params=params).numpy()
    out = dict(omega=og, eigvals=S.lam, K={})
    for thr in TRUNC:
        W = S.weights_trunc(thr)
        out[f"trunc{thr:.0e}"] = (og - Pg @ W) ** 2
        out["K"][f"trunc{thr:.0e}"] = S.n_above(thr)
    for r in RIDGE:
        W = S.weights_ridge(r * S.lam[0])
        out[f"ridge{r:.0e}"] = (og - Pg @ W) ** 2
    # P échantillons i.i.d. (pinv tronquée 1e-4 sur le spectre empirique)
    rng_all = {}
    for P in P_list:
        Es = []
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            oms = render.sample_omegas("creneau", wmax, P, rng)
            Phis = render.make_videos(oms, name, d=d, T=T, method=method, ss=ss, params=params).numpy()
            Ss = spectra.GramSpectrum(Phis, np.ones(P) / P, oms)
            W = Ss.weights_trunc(1e-4)
            Es.append((og - Pg @ W) ** 2)
        out[f"P{P}_trunc1e-04"] = np.mean(Es, 0)
        rng_all[P] = list(seeds)
    out["seeds"] = rng_all
    return out


def summarize(out, wmax):
    og = out["omega"]
    gap = (og > wmax / 3) & (og < 2 * wmax / 3)
    res = {}
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.shape == og.shape and k != "omega":
            res[k] = dict(gap_max=float(v[gap].max()), gap_mean=float(v[gap].mean()),
                          out_max=float(v[~gap].max()), out_mean=float(v[~gap].mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmax", type=float, default=4.0, help="multiple de pi/T")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--method", default="analytic")
    ap.add_argument("--ss", type=int, default=4)
    ap.add_argument("--templates", default="gaussian,segment,quarter_circle,cross,dipole")
    a = ap.parse_args()
    setup()
    d, T = 28, 10
    wmax = a.wmax * np.pi / T
    tag = f"wmax{a.wmax:g}piT_{a.method}{a.ss}"
    names = a.templates.split(",")
    allres = {}
    for n in names:
        f = RES / f"a5_gap_{n}_{tag}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            out = {k: z[k] for k in z.files}
            out["K"] = out["K"].item()
        else:
            out = gap_curves(n, wmax, a.n, d, T, a.method, a.ss)
            np.savez(f, **{k: (v if not isinstance(v, dict) else np.array(v, dtype=object)) for k, v in out.items()})
        allres[n] = out
        s = summarize(out, wmax)
        print(n, json.dumps({k: f"{v['gap_max']:.2e}/{v['out_max']:.2e}" for k, v in s.items()}), flush=True)

    # ---- tableau
    lines = [f"# Expérience de gap — rendu {a.method} ss={a.ss}, ω_max = {a.wmax:g}π/T = {wmax:.4f}, "
             f"gap = [{wmax/3:.3f}, {2*wmax/3:.3f}]\n",
             "Valeurs : max de E(ω) dans le gap / moyenne dans le gap / max hors gap.\n",
             "| template | régresseur | K | max gap | moy. gap | max hors gap |", "|---|---|---|---|---|---|"]
    for n in names:
        s = summarize(allres[n], wmax)
        for k, v in s.items():
            K = allres[n]["K"].get(k, "")
            lines.append(f"| {n} | {k} | {K} | {v['gap_max']:.2e} | {v['gap_mean']:.2e} | {v['out_max']:.2e} |")
    (RES / f"a5_gap_summary_{tag}.md").write_text("\n".join(lines) + "\n")

    # ---- figure format mémoire (trunc 1e-4)
    show = [n for n in ["gaussian", "segment", "quarter_circle", "cross"] if n in names]
    fig, axs = plt.subplots(3, len(show), figsize=(3.3 * len(show), 7.5),
                            gridspec_kw=dict(height_ratios=[0.55, 1, 1]))
    for j, n in enumerate(show):
        out = allres[n]
        V = render.make_videos([0.3], n, d=d, T=T, method=a.method, ss=a.ss, flat=False)[0].numpy()
        axs[0, j].imshow(np.concatenate([V[0], V[2], V[4]], axis=1), cmap="gray", vmin=0, vmax=1)
        axs[0, j].set_title(f"Mouvement : {templates.LABELS_FR[n]} (t=0, 2, 4)", fontsize=8)
        axs[0, j].axis("off")
        lam = out["eigvals"]
        K = out["K"]["trunc1e-04"]
        axs[1, j].hist(np.log10(lam[:K]), bins=30, density=True, color=TEMPLATE_COLOR[n],
                       edgecolor="white", linewidth=0.6)
        axs[1, j].set_title(f"Densité spectrale (K = {K})", fontsize=8)
        axs[1, j].set_xlabel("log10 λ")
        og = out["omega"]
        axs[2, j].axvspan(wmax / 3, 2 * wmax / 3, color=GAP_SHADE, alpha=0.12, lw=0)
        axs[2, j].semilogy(og, np.maximum(out["trunc1e-04"], 1e-12), color="#8b1a1a", lw=1.3)
        axs[2, j].set_ylim(1e-7, 10)
        axs[2, j].set_title("Erreur de test E(ω) (pinv tronquée 1e-4)", fontsize=8)
        axs[2, j].set_xlabel("ω")
    axs[1, 0].set_ylabel("densité")
    axs[2, 0].set_ylabel("(ω − W·φ(ω))²")
    save(fig, FIG / f"a5_gap_memoir_{tag}.png", dict(wmax=wmax, n=a.n, method=a.method, ss=a.ss,
                                                      threshold=1e-4, dist="creneau", d=d, T=T))

    # ---- figure régresseurs
    fig, axs = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4), sharey=True)
    keys = [f"trunc{t:.0e}" for t in TRUNC] + [f"ridge{r:.0e}" for r in RIDGE] + ["P2000_trunc1e-04"]
    labs = ["pinv 1e-4", "pinv 1e-8", "pinv 1e-12", "ridge 1e-3 λ₁", "ridge 1e-6 λ₁", "ridge 1e-9 λ₁",
            "P=2000, pinv 1e-4"]
    styles = ["-", "-", "-", "--", "--", "--", ":"]
    cols = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[0], PALETTE[1], PALETTE[2], "#0b0b0b"]
    for ax, n in zip(axs, names):
        out = allres[n]
        og = out["omega"]
        ax.axvspan(wmax / 3, 2 * wmax / 3, color=GAP_SHADE, alpha=0.12, lw=0)
        for k, l, st, c in zip(keys, labs, styles, cols):
            ax.semilogy(og, np.maximum(out[k], 1e-16), ls=st, color=c, lw=1.1, label=l)
        ax.set_title(templates.LABELS_FR[n], fontsize=9)
        ax.set_xlabel("ω")
        ax.set_ylim(1e-14, 10)
    axs[0].set_ylabel("E(ω)")
    axs[-1].legend(fontsize=6, loc="lower right")
    save(fig, FIG / f"a5_gap_regressors_{tag}.png", dict(wmax=wmax, keys=keys, labels=labs))


if __name__ == "__main__":
    main()
