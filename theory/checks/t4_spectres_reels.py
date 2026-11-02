"""T4 — Quantités spectrales du gap sur les vrais templates (data/spectra, rendu analytic ss=4, distribution créneau).

Pour chaque template et ω_max : K_export, part de l'énergie test captée, E_OOD de la pinv tronquée (P = ∞) pour plusieurs
seuils τ (λ_k/λ_1 > τ), erreur d'entraînement population correspondante, spectre généralisé μ (Σ_test v = μ Σ_train v :
nombre de directions « concentrées dans le gap »), et levier r_k = (Σ_test)_kk/λ_k des modes d'entraînement.
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

TEMPL = ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]
TAUS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


def load(tpl, wtag, suffix=""):
    return np.load(os.path.join(ROOT, "data", "spectra", f"{tpl}_creneau_d28_T10_wmax{wtag}{suffix}.npz"))


def analyse(tpl, wtag, verbose=True):
    d = load(tpl, wtag)
    lam, c = d["eigvals"], d["c_proj"]
    S = d["test_cov_in_train_basis"]; K = S.shape[0]
    ct, o2t = d["test_c_proj"], float(d["test_omega2"])
    wmax = float(d["wmax"]); var_gap = (wmax / 3) ** 2 / 12
    frac = np.trace(S) / float(d["test_trace"])
    ref = float(d["ref_Eood_pop_trunc1e-4"])
    Eo = [th.ood_error_truncated(lam, c, S, ct, o2t, tau=t) for t in TAUS]
    t_k = c[:K] ** 2 / lam[:K]
    Etr = [float(d["omega2"]) - np.sum(t_k[lam[:K] > t * lam[0]]) for t in TAUS]
    mu = th.generalized_ratio_spectrum(lam[:K], S, eig_tol=1e-13)
    r = np.diag(S) / lam[:K]
    out = dict(tpl=tpl, wmax=wmax, K=K, frac=frac, ref=ref, Eo=Eo, Etr=Etr, mu=mu, r=r, t=t_k, var_gap=var_gap,
               lam=lam[:K], S=S, ct=ct, o2t=o2t, c=c[:K], omega2=float(d["omega2"]))
    if verbose:
        print(f"{tpl:>15} ωmax={wmax:.4f} K={K:4d} frac_test={frac:.6f} ref1e-4={ref:.3e} (Var_gap={var_gap:.2e})")
        print("      E_OOD(τ) : " + "  ".join(f"{t:.0e}:{e:.2e}" for t, e in zip(TAUS, Eo)))
        print("      E_tr (τ) : " + "  ".join(f"{t:.0e}:{max(e,0):.1e}" for t, e in zip(TAUS, Etr)))
        print(f"      #μ>1={np.sum(mu>1)}  #μ>1e2={np.sum(mu>1e2)}  #μ>1e4={np.sum(mu>1e4)}  #μ>1e8={np.sum(mu>1e8)}  μ_max={mu[0]:.2e}")
    return out


if __name__ == "__main__":
    for wtag in ["0.6283", "1.2566"]:
        for tpl in TEMPL:
            analyse(tpl, wtag)
        print()
