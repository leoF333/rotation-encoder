"""A1 — figures de génération des données.

- figures/a1_fig2_repro.png : reproduction de la Fig. 2 (gaussienne t = 0, 2, 4 ; template croix)
- figures/a1_templates.png : galerie des 7 templates (t = 0, 2, 4, 6 ; omega = 0.3)
- figures/a1_rendering.png : comparaison des rendus (analytic ss=1, analytic ss=4, interp) pour la croix,
  et intensité d'un pixel en fonction de omega (continuité en omega).
Usage : python scripts/a1_data_figures.py
"""
import sys
from pathlib import Path

import numpy as np

from aerot import render, templates
from aerot.paths import FIG
from aerot.plotting import setup, save, plt, TEMPLATE_COLOR

setup()
d, T = 28, 10

# ---- Fig. 2 du mémoire
om = 0.29
V = render.make_videos([om], "gaussian", d=d, T=T, method="interp", flat=False)[0].numpy()
C = render.raster("cross", d).numpy()
fig, axs = plt.subplots(1, 4, figsize=(8, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.0]))
for i, t in enumerate([0, 2, 4]):
    axs[i].imshow(V[t], cmap="gray", vmin=0, vmax=1)
    axs[i].set_title(f"gaussienne, t = {t}")
axs[3].imshow(C, cmap="gray", vmin=0, vmax=1)
axs[3].set_title("template « croix »")
for a in axs:
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle(f"Reproduction de la Fig. 2 (ω = {om} rad/image, d = {d})", y=1.04)
save(fig, FIG / "a1_fig2_repro.png", dict(omega=om, d=d, T=T, method="interp",
                                          gaussian=templates.template_params("gaussian", d),
                                          cross=templates.template_params("cross", d)))

# ---- galerie
names = templates.TEMPLATE_NAMES
ts = [0, 2, 4, 6]
fig, axs = plt.subplots(len(names), len(ts), figsize=(1.6 * len(ts), 1.6 * len(names)))
for i, n in enumerate(names):
    V = render.make_videos([0.3], n, d=d, T=T, method="interp", flat=False)[0].numpy()
    for j, t in enumerate(ts):
        ax = axs[i, j]
        ax.imshow(V[t], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i == 0:
            ax.set_title(f"t = {t}")
        if j == 0:
            ax.set_ylabel(templates.LABELS_FR[n], fontsize=8)
fig.suptitle("Templates (rendu interp, ω = 0.3 rad/image)", y=1.0)
save(fig, FIG / "a1_templates.png", dict(omega=0.3, d=d, T=T, method="interp",
                                         params={n: templates.template_params(n, d) for n in names}))

# ---- comparaison des rendus
meths = [("analytic", 1, "analytique ss=1"), ("analytic", 4, "analytique ss=4"), ("interp", 1, "interp (bilinéaire)")]
fig = plt.figure(figsize=(9, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
for j, (m, ss, lab) in enumerate(meths):
    V = render.make_videos([0.3], "cross", d=d, T=T, method=m, ss=ss, flat=False)[0].numpy()
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(V[3], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"croix, t = 3 — {lab}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ws = np.linspace(0, 4 * np.pi / T, 3000)
ax = fig.add_subplot(gs[1, :])
pix = (3, 7, 18)  # (t, ligne, colonne)
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
for (m, ss, lab), c in zip(meths, cols):
    V = render.make_videos(ws, "cross", d=d, T=T, method=m, ss=ss, flat=False).numpy()
    ax.plot(ws, V[:, pix[0], pix[1], pix[2]], color=c, lw=1.2, label=lab)
ax.set_xlabel("ω (rad/image)")
ax.set_ylabel(f"intensité du pixel (t={pix[0]}, ligne {pix[1]}, col. {pix[2]})")
ax.legend(loc="upper right")
ax.set_title("φ(ω) pour un pixel : constante par morceaux (analytique) vs continue (interp)")
save(fig, FIG / "a1_rendering.png", dict(template="cross", omega_img=0.3, t_img=3, pixel=pix,
                                         methods=meths, d=d, T=T))
print("ok")
