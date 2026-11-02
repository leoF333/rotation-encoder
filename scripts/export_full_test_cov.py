"""Matrices test (gap) complètes dans la base propre d'entraînement (créneau), sur TOUS les modes λ_k > 1e-14 λ_1
(l'export data/spectra les tronque à 1000 modes).

Sortie : results/b1_theory_cache/<template>_creneau_wmax<ω_max>_n4000_tol1e-14.npz, clés lam, c (u_k·C), omega2,
test_cov (u_kᵀ Σ_test u_l), test_c (u_k·C_test), test_omega2, test_trace (Tr Σ_test). Fichiers de 10–25 Mo, non commités.
Usage : python scripts/export_full_test_cov.py [--wmax 2] [--templates gaussian,quarter_circle,cross,segment]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from aerot import render, spectra
from aerot.paths import RES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmax", type=float, default=2.0, help="multiple de π/T")
    ap.add_argument("--templates", default="gaussian,quarter_circle,cross,segment,dipole")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    T, tol = 10, 1e-14
    wmax = a.wmax * np.pi / T
    out = RES / "b1_theory_cache"
    out.mkdir(exist_ok=True)
    for name in a.templates.split(","):
        f = out / f"{name}_creneau_wmax{wmax:.4f}_n{a.n}_tol{tol:.0e}.npz"
        if f.exists():
            continue
        om, w = render.quadrature("creneau", wmax, a.n)
        S = spectra.GramSpectrum(render.make_videos(om, name, d=28, T=T).numpy(), w, om)
        K = int(np.sum(S.lam > tol * S.lam[0]))
        og, wg = render.quadrature("gap", wmax, a.n_test)
        tq = S.test_quantities(render.make_videos(og, name, d=28, T=T).numpy(), wg, og, K)
        np.savez(f, lam=S.lam[:K], c=S.c_proj[:K], omega2=S.omega2, test_cov=tq["test_cov_in_train_basis"],
                 test_c=tq["test_c_proj"], test_omega2=tq["test_omega2"], test_trace=tq["test_trace"])
        print(f.name, "K =", K, "fraction d'énergie test couverte =",
              float(np.trace(tq["test_cov_in_train_basis"]) / tq["test_trace"]), flush=True)
