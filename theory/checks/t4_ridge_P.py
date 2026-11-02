"""T4 — Erreur dans le gap à P fini avec ridge (formule des répliques theory.ood_error) sur les vrais spectres.

Pour chaque template (créneau, ω_max donné) : E_OOD(P, λ_r) sur une grille de κ_0 = λ_r/P (en unités de λ_1),
ridge optimal pour le gap vs ridge optimal en distribution (E_gen sur le créneau), décomposition biais/variance,
facteur d'amplification de la variance A(κ) = Σ_k r_k L_k² / Σ_k L_k², r_k = (Σ_test)_kk/λ_k (levier des modes).
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aerot import theory as th
from t4_spectres_reels import load

GRID = 10.0 ** np.arange(-14, -1.9, 0.25)     # κ0/λ_1


def pop_bias(lam, c, S, ct, o2t, kap):
    K = S.shape[0]
    W = c[:K] / (lam[:K] + kap)
    return float(o2t - 2 * ct[:K] @ W + W @ S @ W)


def scan(tpl, wtag, P_list=(100, 1000, 10000), verbose=True):
    d = load(tpl, wtag)
    lam, c, om2 = d["eigvals"], d["c_proj"], float(d["omega2"])
    S, ct, o2t = d["test_cov_in_train_basis"], d["test_c_proj"], float(d["test_omega2"])
    K = S.shape[0]
    keep = lam > 1e-15 * lam[0]
    lamk, ck = lam[keep], c[keep]
    r = np.diag(S) / lam[:K]
    res = {}
    # P = ∞ : biais pur, κ = κ0
    Einf = np.array([pop_bias(lam, c, S, ct, o2t, g * lam[0]) for g in GRID])
    Ginf = np.array([om2 - np.sum(ck**2 * (lamk + 2 * g * lam[0]) / (lamk + g * lam[0]) ** 2) for g in GRID])
    res["inf"] = (Einf, Ginf)
    for P in P_list:
        Eo, Eg, Av = [], [], []
        for g in GRID:
            lr = g * lam[0] * P
            o = th.ood_error(lamk, ck, om2, S, ct, o2t, P, lr, return_parts=True)
            Eo.append(o["Eood"]); Eg.append(o["Egen_train"])
            L = lam[:K] / (lam[:K] + o["kappa"])
            Av.append(np.sum(r * L**2) / np.sum(L**2))
        res[P] = (np.array(Eo), np.array(Eg), np.array(Av))
    if verbose:
        print(f"{tpl} ωmax={float(d['wmax']):.4f}")
        for key in ["inf"] + list(P_list):
            Eo, Eg = res[key][0], res[key][1]
            io, ig = np.argmin(Eo), np.argmin(Eg)
            extra = f"  A(κ*_OOD)={res[key][2][io]:.1e}" if key != "inf" else ""
            print(f"  P={str(key):>6}: min E_OOD={Eo[io]:.2e} en κ0={GRID[io]:.0e} | min E_gen(créneau)={max(Eg[ig],0):.1e} en κ0={GRID[ig]:.0e}"
                  f" ; E_OOD au κ*_gen = {Eo[ig]:.2e}{extra}")
    return res


if __name__ == "__main__":
    for wtag in ["0.6283", "1.2566"]:
        for tpl in ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]:
            scan(tpl, wtag)
