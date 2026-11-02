"""A4 — autoencodeurs (Fig. 3 et Fig. 4 bas du mémoire), template arc de cercle.

Architectures (latent k = 1) :
- sym3 : N -> 1 -> N (encodeur linéaire, décodeur 1 couche + sigmoïde)
- sym5 : N -> 128 -> 1 -> 128 -> N (ReLU)
- asym : encodeur linéaire N -> 1, décodeur profond 1 -> 64 -> 256 -> 1024 -> N (ReLU)
Données : omega ~ U[0.05, 2pi/T] (Fig. 4), P = 4000 (entraînement), 1000 (test), rendu par défaut.
Entraînement : Adam lr 1e-3, cosinus, 300 époques, batch 128, MSE ; seeds 0, 1, 2.
Sorties : results/a4/<arch>_s<seed>.json (+ .npz latents), results/a4_summary.md,
figures/a4_fig3_repro.png, figures/a4_fig4_latent.png.
Idempotent (saute les runs déjà faits). Usage : python scripts/a4_autoencoders.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from aerot import render, ae, fits
from aerot.render import get_device
from aerot.paths import FIG, RES
from aerot.plotting import setup, save, plt, PALETTE

D, T = 28, 10
N = D * D * T
WLO, WHI = 0.05, 2 * np.pi / T
TEMPLATE = "arc"
ARCHS = ["sym3", "sym5", "asym"]
SEEDS = [0, 1, 2]
EPOCHS = 300
OUT = RES / "a4"


def data(dev):
    rtr = np.random.default_rng(0)
    rte = np.random.default_rng(1)
    om_tr = rtr.uniform(WLO, WHI, 4000)
    om_te = np.sort(rte.uniform(WLO, WHI, 1000))
    Xtr = render.make_videos(om_tr, TEMPLATE, d=D, T=T, device=dev, dtype=torch.float32)
    Xte = render.make_videos(om_te, TEMPLATE, d=D, T=T, device=dev, dtype=torch.float32)
    return om_tr, Xtr, om_te, Xte


def run(arch, seed, dev, om_tr, Xtr, om_te, Xte):
    f = OUT / f"{arch}_s{seed}.json"
    if f.exists():
        return json.loads(f.read_text())
    torch.manual_seed(seed)
    model = ae.build(arch, N, k=1)
    info = ae.train(model, Xtr, epochs=EPOCHS, batch=128, lr=1e-3, device=dev, seed=seed, log_every=25)
    z_te, e_te = ae.encode_eval(model, Xte, device=dev)
    z_tr, e_tr = ae.encode_eval(model, Xtr, device=dev)
    z1 = z_te.reshape(-1)
    met = ae.latent_metrics(z1, om_te)
    r2s, _, psin = fits.fit_sin(om_te, z1, bmax=30)
    ylin, _ = fits.fit_lin(om_te, z1)
    x194 = render.make_videos([0.194], TEMPLATE, d=D, T=T, device=dev, dtype=torch.float32)
    rec, _ = ae.reconstruct(model, x194, device=dev)
    res = dict(arch=arch, seed=seed, template=TEMPLATE, n_params=ae.n_params(model), epochs=EPOCHS,
               mse_test=float(e_te.mean()), mse_train=float(e_tr.mean()), train_time_s=info["train_time_s"],
               history=info["history"], R2_lin_z=fits.r2(z1, ylin), R2_sin_z=float(r2s), **met)
    np.savez(OUT / f"{arch}_s{seed}.npz", z_test=z1, omega_test=om_te, rec_w0194=rec.reshape(T, D, D),
             err_test=e_te)
    f.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
    return res


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    om_tr, Xtr, om_te, Xte = data(dev)
    allres = {a: [run(a, s, dev, om_tr, Xtr, om_te, Xte) for s in SEEDS] for a in ARCHS}

    keys = ["mse_test", "spearman_abs", "R2_monotone", "R2_knn", "R2_lin_z", "R2_sin_z", "n_params", "train_time_s"]
    lines = ["| architecture | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for a in ARCHS:
        vals = []
        for k in keys:
            v = np.array([r[k] for r in allres[a]], float)
            vals.append(f"{v.mean():.4g} ± {v.std():.2g}" if k not in ("n_params",) else f"{int(v[0])}")
        lines.append(f"| {a} | " + " | ".join(vals) + " |")
    txt = ("# Autoencodeurs — template arc, ω ~ U[0.05, 2π/T], latent k = 1, 3 seeds (moyenne ± écart-type)\n\n"
           + "\n".join(lines) + "\n\nmse = moyenne par pixel sur le test ; R2_monotone = régression isotone de ω sur z ; "
           "R2_knn = 5-NN leave-one-out ; R2_lin_z / R2_sin_z = ajustements de z(ω) comme la Fig. 4 "
           "(mémoire : 0.6409 / 0.9061).\n")
    (RES / "a4_summary.md").write_text(txt)
    print(txt)

    # Fig. 3 : reconstructions à omega = 0.194
    x = render.make_videos([0.194], TEMPLATE, d=D, T=T, flat=False)[0].numpy()
    fig, axs = plt.subplots(4, T, figsize=(1.25 * T, 5.6))
    rows = [("original", x), ("AE 3 couches", np.load(OUT / "sym3_s0.npz")["rec_w0194"]),
            ("original", x), ("AE 5 couches", np.load(OUT / "sym5_s0.npz")["rec_w0194"])]
    for i, (lab, v) in enumerate(rows):
        for t in range(T):
            axs[i, t].imshow(v[t], cmap="gray", vmin=0, vmax=1)
            axs[i, t].set_xticks([]); axs[i, t].set_yticks([]); axs[i, t].grid(False)
            if t == 0:
                axs[i, t].set_ylabel(lab, fontsize=7)
            if i in (0, 2):
                axs[i, t].set_title(f"t={t}", fontsize=7)
    fig.suptitle("Reproduction de la Fig. 3 : reconstruction de l'arc (ω = 0.194), AE 3 vs 5 couches (seed 0)")
    save(fig, FIG / "a4_fig3_repro.png", dict(omega=0.194, archs=["sym3", "sym5"], seed=0, epochs=EPOCHS))

    # Fig. 4 bas : latent vs omega
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, a in zip(axs, ARCHS):
        z = np.load(OUT / f"{a}_s0.npz")
        zz, oo = z["z_test"], z["omega_test"]
        ylin, _ = fits.fit_lin(oo, zz)
        r2s, ys, _ = fits.fit_sin(oo, zz, bmax=30)
        r = allres[a][0]
        ax.scatter(oo, zz, s=5, color=PALETTE[0], alpha=0.4, lw=0, label="latent z")
        ax.plot(oo, ylin, color="#e34948", lw=1.3, label=f"linéaire (R²={fits.r2(zz, ylin):.4f})")
        ax.plot(oo, ys, color="#008300", lw=1.3, label=f"sinusoïdal (R²={r2s:.4f})")
        ax.set_title(f"{a} — |ρ_S| = {r['spearman_abs']:.4f}, R²_mono = {r['R2_monotone']:.4f}", fontsize=8)
        ax.set_xlabel("ω (rad/image)")
        ax.set_ylabel("z")
        ax.legend(fontsize=6)
    save(fig, FIG / "a4_fig4_latent.png", dict(template=TEMPLATE, seed=0, archs=ARCHS))


if __name__ == "__main__":
    main()
