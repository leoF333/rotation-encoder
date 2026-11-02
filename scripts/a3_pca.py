"""A3 — PCA (Sec. 2.1, Fig. 4 haut du mémoire).

Données : omega ~ U[0.05, 2pi/T] (axe de la Fig. 4), n = 500 vidéos, PCA centrée.
Pour chaque template : projection sur PC1 vs omega, R^2 d'un ajustement linéaire et sinusoïdal
(a sin(b omega + c) + e, moindres carrés non linéaires, meilleur de plusieurs initialisations),
nombre de changements de monotonie (bijectivité).
Sorties : results/a3_pca.json, results/a3_pca.md, figures/a3_pca.png, figures/a3_pca_fig4_repro.png
Usage : python scripts/a3_pca.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from aerot import render, spectra, templates
from aerot.paths import FIG, RES
from aerot.plotting import setup, save, plt, TEMPLATE_COLOR, PALETTE


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_lin(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


def fit_sin(x, y):
    """a sin(b x + c) + e : balayage fin de b (moindres carrés linéaires en (a cos c, a sin c, e)),
    puis raffinement non linéaire de tous les paramètres."""
    f = lambda x, a, b, c, e: a * np.sin(b * x + c) + e  # noqa: E731
    best_b, best_s = None, -np.inf
    for b in np.linspace(0.5, 60, 6000):
        A = np.vstack([np.sin(b * x), np.cos(b * x), np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        s = r2(y, A @ coef)
        if s > best_s:
            best_s, best_b, best_coef = s, b, coef
    a0 = np.hypot(best_coef[0], best_coef[1])
    c0 = np.arctan2(best_coef[1], best_coef[0])
    p0 = [a0, best_b, c0, best_coef[2]]
    try:
        p, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
        if r2(y, f(x, *p)) < best_s:
            p = p0
    except RuntimeError:
        p = p0
    return r2(y, f(x, *p)), f(x, *p), p


def monotone_changes(x, y, smooth=9):
    o = np.argsort(x)
    ys = np.convolve(y[o], np.ones(smooth) / smooth, mode="valid")
    dy = np.sign(np.diff(ys))
    dy = dy[dy != 0]
    return int(np.sum(dy[1:] != dy[:-1]))


def main():
    setup()
    d, T = 28, 10
    rng = np.random.default_rng(0)
    n = 500
    wlo, whi = 0.05, 2 * np.pi / T
    om = np.sort(rng.uniform(wlo, whi, n))
    names = ["arc", "gaussian", "dipole", "quarter_circle", "cross", "segment"]
    res = {}
    curves = {}
    for name in names:
        for method in ["interp", "analytic"]:
            Phi = render.make_videos(om, name, d=d, T=T, method=method, ss=4 if method == "analytic" else 1).numpy()
            proj, evr, comps = spectra.pca_projection(Phi, 2)
            pc1 = proj[:, 0]
            if np.corrcoef(pc1, om)[0, 1] < 0:
                pc1 = -pc1
            yl, _ = fit_lin(om, pc1)
            s_sin, ys, psin = fit_sin(om, pc1)
            res[f"{name}/{method}"] = dict(R2_lin=float(r2(pc1, yl)), R2_sin=float(s_sin),
                                           sin_params=[float(v) for v in psin], evr1=float(evr[0]),
                                           evr2=float(evr[1]), monotony_changes=monotone_changes(om, pc1),
                                           pc1_range=[float(pc1.min()), float(pc1.max())])
            curves[f"{name}/{method}"] = (pc1, yl, ys)
            print(name, method, json.dumps(res[f"{name}/{method}"]), flush=True)
    (RES / "a3_pca.json").write_text(json.dumps(res, indent=1))
    lines = ["| template | rendu | R² linéaire | R² sinusoïdal | var. expl. PC1 | changements de monotonie | plage PC1 |",
             "|---|---|---|---|---|---|---|",
             "| *mémoire (Fig. 4)* | ? | *0.9261* | *0.9995* | ? | 2 | [−7.8, 7.8] |"]
    for k, v in res.items():
        nm, me = k.split("/")
        lines.append(f"| {nm} | {me} | {v['R2_lin']:.4f} | {v['R2_sin']:.4f} | {v['evr1']:.3f} | "
                     f"{v['monotony_changes']} | [{v['pc1_range'][0]:.1f}, {v['pc1_range'][1]:.1f}] |")
    (RES / "a3_pca.md").write_text("# PCA : PC1 vs ω (ω ~ U[0.05, 2π/T], n = 500)\n\n" + "\n".join(lines) + "\n")

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, name in zip(axs.flat, names):
        pc1, yl, ys = curves[f"{name}/interp"]
        v = res[f"{name}/interp"]
        ax.scatter(om, pc1, s=6, color=TEMPLATE_COLOR[name], alpha=0.35, lw=0, label="données (PC1)")
        ax.plot(om, yl, color="#e34948", lw=1.4, label=f"linéaire (R²={v['R2_lin']:.4f})")
        ax.plot(om, ys, color="#008300", lw=1.4, label=f"sinusoïdal (R²={v['R2_sin']:.4f})")
        ax.set_title(templates.LABELS_FR[name], fontsize=9)
        ax.set_xlabel("ω (rad/image)")
        ax.set_ylabel("PC1")
        ax.legend(fontsize=6, loc="best")
    save(fig, FIG / "a3_pca.png", dict(n=n, omega_range=[wlo, whi], d=d, T=T, method="interp", seed=0))
    print("ok")


if __name__ == "__main__":
    main()
