"""A2 — analyse des spectres (à lancer après a2_export_spectra.py).

Produit :
- results/a2_K_table.md : K (seuils 1e-4, 1e-8, 1e-12 ; non centré / centré), modes à 99.9 % d'énergie, lambda_1
- figures/a2_spectra.png : lambda_k / lambda_1 vs k (échelles semi-log et log-log), rendu analytic ss=4 (défaut)
- figures/a2_densities.png : densités spectrales en log10 lambda (style Fig. 6-7 du mémoire)
- figures/a2_rendering_spectra.png : effet du rendu (analytic ss=1,4,8 / interp) sur le spectre
- results/a2_symmetry.md : croix vs quart de cercle vs bras seul (critères de comptage)
- figures/a2_jacobi_anger.png : coefficients de Fourier angulaires de la gaussienne vs A(r) I_k(z(r)),
  et spectre de puissance angulaire E_k des templates
Usage : python scripts/a2_spectra_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.special as sps
import torch

from aerot import render, spectra, templates
from aerot.paths import FIG, RES, SPECTRA
from aerot.plotting import setup, save, plt, TEMPLATE_COLOR, PALETTE

setup()
d, T = 28, 10
WM = {"2π/T": 2 * np.pi / T, "4π/T": 4 * np.pi / T}
NAMES = templates.TEMPLATE_NAMES


def load(name, dist, wmax, suffix=""):
    return np.load(SPECTRA / f"{name}_{dist}_d{d}_T{T}_wmax{wmax:.4f}{suffix}.npz")


def n_energy(lam, frac=0.999):
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)


# ------------------------------------------------------------------ 1. table des K
rows = ["| template | ρ | ω_max | λ_1 | K(1e-4) | K(1e-8) | K(1e-12) | K_c(1e-4) | K_c(1e-12) | modes 99.9 % (non centré) | modes 99.9 % (centré) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for wl, wmax in WM.items():
    for dist in ["uniform", "creneau"]:
        for n in NAMES:
            z = load(n, dist, wmax)
            lam, lc = z["eigvals"], z["eigvals_centered"]
            rows.append(f"| {n} | {dist} | {wl} | {lam[0]:.3g} | {int(z['K_1e4'])} | {int(z['K_1e8'])} | "
                        f"{int(z['K_1e12'])} | {int(z['K_1e4_centered'])} | {int(z['K_1e12_centered'])} | "
                        f"{n_energy(lam)} | {n_energy(lc)} |")
note = ("\n\nRendu `analytic` ss=4 (défaut), quadrature de Gauss-Legendre n = 4000 (rang ≤ 4000 : les K(1e-12) proches de 4000 "
        "sont limités par la quadrature). K(seuil) = #{k : λ_k/λ_1 > seuil}. Pour le disque (invariant par rotation), "
        "tout mode au-delà du premier serait un artefact de rendu (ici λ_2/λ_1 ~ 1e-15).\n")
(RES / "a2_K_table.md").write_text("# Nombre de modes par template (population)\n\n" + "\n".join(rows) + note)
print("\n".join(rows))

# ------------------------------------------------------------------ 2. spectres
fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
for j, (wl, wmax) in enumerate(WM.items()):
    for n in NAMES:
        z = load(n, "uniform", wmax)
        lam = z["eigvals"] / z["eigvals"][0]
        k = np.arange(1, len(lam) + 1)
        m = lam > 1e-15
        axs[0, j].semilogy(k[m][:400], lam[m][:400], color=TEMPLATE_COLOR[n], label=templates.LABELS_FR[n], lw=1.3)
        axs[1, j].loglog(k[m], lam[m], color=TEMPLATE_COLOR[n], label=templates.LABELS_FR[n], lw=1.3)
    for i in range(2):
        axs[i, j].axhline(1e-4, color="#52514e", ls="--", lw=0.8)
        axs[i, j].axhline(1e-12, color="#52514e", ls=":", lw=0.8)
        axs[i, j].set_ylim(1e-16, 2)
        axs[i, j].set_xlabel("rang k")
        axs[i, j].set_ylabel("λ_k / λ_1")
    axs[0, j].set_title(f"ρ uniforme, ω_max = {wl} (semi-log)")
    axs[1, j].set_title(f"ρ uniforme, ω_max = {wl} (log-log)")
axs[0, 0].legend(ncol=2, loc="upper right")
save(fig, FIG / "a2_spectra.png", dict(d=d, T=T, method="analytic", ss=4, n=4000, dist="uniform",
                                       thresholds=[1e-4, 1e-12]))

# ------------------------------------------------------------------ 3. densités (style mémoire)
fig, axs = plt.subplots(2, 4, figsize=(13, 5.5))
for j, n in enumerate(["gaussian", "segment", "quarter_circle", "cross"]):
    for i, wl in enumerate(WM):
        z = load(n, "creneau", WM[wl])
        lam = z["eigvals"]
        K = int(z["K_1e4"])
        axs[i, j].hist(np.log10(lam[:K]), bins=30, density=True, color=TEMPLATE_COLOR[n],
                       edgecolor="white", linewidth=0.6)
        axs[i, j].set_title(f"{templates.LABELS_FR[n]} — K = {K} (créneau, ω_max = {wl})", fontsize=8)
        axs[i, j].set_xlabel("log10 λ")
axs[0, 0].set_ylabel("densité")
axs[1, 0].set_ylabel("densité")
fig.suptitle("Densités spectrales des modes retenus (λ_k/λ_1 > 1e-4), à comparer aux Fig. 6–7 du mémoire "
             "(gaussienne ≈7 modes, segment ≈37, quart de cercle K = 37, croix K = 45)", fontsize=9)
save(fig, FIG / "a2_densities.png", dict(dist="creneau", threshold=1e-4, bins=30, method="analytic", ss=4))

# ------------------------------------------------------------------ 4. effet du rendu
wmax = WM["4π/T"]
om, w = render.quadrature("uniform", wmax, 2000)
variants = [("interp", 1), ("analytic", 1), ("analytic", 4), ("analytic", 8)]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
rend_rows = ["| template | rendu | K(1e-4) | K(1e-8) | K(1e-12) |", "|---|---|---|---|---|"]
for ax, n in zip(axs, ["gaussian", "quarter_circle", "cross"]):
    for (m, ss), c in zip(variants, PALETTE):
        Phi = render.make_videos(om, n, method=m, ss=ss).numpy()
        S = spectra.GramSpectrum(Phi, w, om)
        lam = S.lam / S.lam[0]
        k = np.arange(1, len(lam) + 1)
        msk = lam > 1e-16
        ax.loglog(k[msk], lam[msk], color=c, lw=1.2, label=f"{m} ss={ss}" if m == "analytic" else "interp")
        rend_rows.append(f"| {n} | {m} ss={ss} | {S.n_above(1e-4)} | {S.n_above(1e-8)} | {S.n_above(1e-12)} |")
    ax.set_title(f"{templates.LABELS_FR[n]} (ρ uniforme, ω_max = 4π/T, n = 2000)", fontsize=8)
    ax.set_xlabel("rang k")
    ax.set_ylim(1e-17, 2)
axs[0].set_ylabel("λ_k / λ_1")
axs[0].legend()
save(fig, FIG / "a2_rendering_spectra.png", dict(variants=variants, dist="uniform", wmax=wmax, n=2000))

# ------------------------------------------------------------------ 5. symétrie : croix vs quart de cercle vs bras
sym_rows = ["| template | ω_max | ρ | rendu | K(1e-4) | K(1e-12) | K_c(1e-4) | 99.9 % (non c.) | 99.9 % (c.) | 99 % (c.) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
for wl, wmax in WM.items():
    for dist in ["uniform", "creneau"]:
        om, w = render.quadrature(dist, wmax, 2000)
        for n in ["quarter_circle", "cross", "arm"]:
            for m, ss in [("interp", 1), ("analytic", 4)]:
                Phi = render.make_videos(om, n, method=m, ss=ss).numpy()
                S = spectra.GramSpectrum(Phi, w, om)
                Sc = spectra.GramSpectrum(Phi, w, om, centered=True)
                sym_rows.append(f"| {n} | {wl} | {dist} | {m} ss={ss} | {S.n_above(1e-4)} | {S.n_above(1e-12)} | "
                                f"{Sc.n_above(1e-4)} | {n_energy(S.lam)} | {n_energy(Sc.lam)} | "
                                f"{n_energy(Sc.lam, 0.99)} |")
(RES / "a2_symmetry.md").write_text(
    "# Croix vs quart de cercle vs bras seul\n\n" + "\n".join(sym_rows)
    + "\n\n## Effet du rendu (ρ uniforme, ω_max = 4π/T, n = 2000)\n\n" + "\n".join(rend_rows) + "\n")
print("\n".join(sym_rows))
print("\n".join(rend_rows))

# ------------------------------------------------------------------ 6. Jacobi-Anger et spectre angulaire
nth = 2048
th = torch.linspace(0, 2 * np.pi, nth + 1, dtype=torch.float64)[:-1]
rs = np.linspace(0.05, 19.5, 400)


def angular_coeffs(name, params=None):
    p = templates.template_params(name, d) if params is None else params
    R = torch.as_tensor(rs)[:, None]
    X = R * torch.cos(th)[None, :]
    Y = R * torch.sin(th)[None, :]
    vals = templates.evaluate(name, X, Y, **p).numpy()
    return np.fft.rfft(vals, axis=1) / nth  # a_k(r), k >= 0


pg = templates.template_params("gaussian", d)
a = np.abs(angular_coeffs("gaussian"))
sig, r0 = pg["sigma"], pg["r0"]
A = np.exp(-(rs**2 + r0**2) / (2 * sig**2))
zr = rs * r0 / sig**2
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
ks = np.arange(0, 40)
for ri, c in zip([40, 164, 300], PALETTE):
    pred = A[ri] * sps.iv(ks, zr[ri])
    axs[0].semilogy(ks, a[ri, ks], "o", ms=3.5, color=c, label=f"r = {rs[ri]:.1f} (numérique)")
    axs[0].semilogy(ks, pred, "-", color=c, lw=1, label=f"A(r) I_k(z(r))")
axs[0].set_ylim(1e-18, 2)
axs[0].set_xlabel("ordre angulaire k")
axs[0].set_ylabel("|a_k(r)|")
axs[0].set_title("Gaussienne : Jacobi-Anger (Éq. 3)")
axs[0].legend(fontsize=6, ncol=1)
relerr = []
for ri in range(len(rs)):
    pred = A[ri] * sps.iv(ks, zr[ri])
    good = pred > 1e-12
    relerr.append(np.max(np.abs(a[ri, ks][good] - pred[good]) / pred[good]))
axs[1].semilogy(rs, relerr, color=PALETTE[0])
axs[1].set_xlabel("r (pixels)")
axs[1].set_ylabel("max_k erreur relative (a_k > 1e-12)")
axs[1].set_title("Écart numérique / formule exacte")
# spectre de puissance angulaire E_k = ∫ |a_k(r)|^2 r dr (k >= 0)
dr = rs[1] - rs[0]
kk = np.arange(1, 400)
for n in ["gaussian", "dipole", "quarter_circle", "cross", "arm", "segment", "arc"]:
    ak = np.abs(angular_coeffs(n)) ** 2
    Ek = (ak * rs[:, None]).sum(0) * dr
    col = TEMPLATE_COLOR.get(n, "#52514e")
    msk = Ek[kk] > 1e-30
    axs[2].loglog(kk[msk], Ek[kk][msk] / Ek[0], color=col, lw=1.1, label=templates.LABELS_FR[n],
                  ls="--" if n == "arm" else "-")
axs[2].loglog(kk, 0.3 * kk**-2.0, color="#52514e", lw=0.8, ls=":", label="∝ k⁻²")
axs[2].loglog(kk, 0.3 * kk**-3.0, color="#52514e", lw=0.8, ls="-.", label="∝ k⁻³")
axs[2].set_ylim(1e-12, 2)
axs[2].set_xlabel("ordre angulaire k")
axs[2].set_ylabel("E_k / E_0")
axs[2].set_title("Spectre de puissance angulaire du template continu")
axs[2].legend(fontsize=6, ncol=2)
save(fig, FIG / "a2_jacobi_anger.png", dict(gaussian=pg, n_theta=nth, r_grid=[0.05, 19.5, 400]))
print("max rel err Jacobi-Anger (r in [1,18]):", max(e for r, e in zip(rs, relerr) if 1 <= r <= 18))
