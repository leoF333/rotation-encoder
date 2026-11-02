"""A2 — export des spectres « population » vers data/spectra/ (format CONVENTIONS.md).

Pour chaque template, distribution (uniform, creneau) et omega_max (2pi/T, 4pi/T) :
second moment non centré Sigma = <phi phi^T>_rho par quadrature de Gauss-Legendre (n noeuds),
spectre via l'astuce de Gram (float64), projections c_k = u_k . C, u_k . phibar, spectre centré,
et pour la distribution créneau les quantités test (gap) dans la base propre d'entraînement.

Idempotent : saute les fichiers déjà présents (sauf --force).
Usage : python scripts/a2_export_spectra.py [--n 4000] [--method analytic] [--ss 4] [--suffix _interp] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aerot import render, spectra, templates
from aerot.paths import SPECTRA, RES


def fname(name, dist, d, T, wmax, suffix=""):
    return SPECTRA / f"{name}_{dist}_d{d}_T{T}_wmax{wmax:.4f}{suffix}.npz"


def export_one(name, dist, d, T, wmax, n, method, ss, suffix, K_cap, n_test, force=False):
    out = fname(name, dist, d, T, wmax, suffix)
    if out.exists() and not force:
        print("déjà fait :", out.name, flush=True)
        return
    t0 = time.time()
    om, w = render.quadrature(dist, wmax, n)
    Phi = render.make_videos(om, name, d=d, T=T, method=method, ss=ss).numpy()
    S = spectra.GramSpectrum(Phi, w, om)
    Sc = spectra.GramSpectrum(Phi, w, om, centered=True)
    lam = S.lam
    K12 = S.n_above(1e-12)
    K_export = int(min(K12, K_cap))
    data = dict(
        eigvals=lam, c_proj=S.c_proj, omega2=S.omega2, omega_mean=S.omega_mean,
        phi_mean_proj=S.phi_mean_proj, N=Phi.shape[1], d=d, T=T, wmax=wmax,
        n_samples_used=len(om), eigvals_centered=Sc.lam, c_proj_centered=Sc.c_proj,
        omega_var=S.omega2 - S.omega_mean**2,
        K_1e4=S.n_above(1e-4), K_1e8=S.n_above(1e-8), K_1e12=K12,
        K_1e4_centered=Sc.n_above(1e-4), K_1e12_centered=Sc.n_above(1e-12),
        trace=lam.sum(),
    )
    # erreurs de référence (population) pour la pinv tronquée 1e-4
    W = S.weights_trunc(1e-4)
    data["ref_Etrain_pop_trunc1e-4"] = float(w @ (om - Phi @ W) ** 2)
    if dist == "creneau":
        omt, wt = render.quadrature("gap", wmax, n_test)
        Phit = render.make_videos(omt, name, d=d, T=T, method=method, ss=ss).numpy()
        tq = S.test_quantities(Phit, wt, omt, K_export)
        data.update(tq)
        data["K_export"] = K_export
        data["n_test_samples_used"] = len(omt)
        data["ref_Eood_pop_trunc1e-4"] = float(wt @ (omt - Phit @ W) ** 2)
    np.savez_compressed(out, **data)
    meta = dict(template=name, params=templates.template_params(name, d), dist=dist, d=d, T=T,
                wmax=wmax, n=len(om), method=method, ss=ss, quadrature="Gauss-Legendre composite q=8",
                K_1e4=data["K_1e4"], K_1e12=K12, lam1=float(lam[0]),
                time_s=round(time.time() - t0, 1))
    print(json.dumps(meta), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--method", default="analytic")
    ap.add_argument("--ss", type=int, default=4)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--K_cap", type=int, default=1000)
    ap.add_argument("--d", type=int, default=28)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--templates", default=",".join(templates.TEMPLATE_NAMES))
    ap.add_argument("--wmax", default="2,4", help="multiples de pi/T")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    SPECTRA.mkdir(parents=True, exist_ok=True)
    metas = []
    for wm in [float(x) for x in a.wmax.split(",")]:
        wmax = wm * np.pi / a.T
        for dist in ["uniform", "creneau"]:
            for name in a.templates.split(","):
                m = export_one(name, dist, a.d, a.T, wmax, a.n, a.method, a.ss, a.suffix,
                               a.K_cap, a.n_test, a.force)
                if m:
                    metas.append(m)
    log = RES / "a2_export_spectra_meta.jsonl"
    with open(log, "a") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    main()
