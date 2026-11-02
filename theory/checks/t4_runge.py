"""T4 — Identité « de Runge » pour l'erreur dans le gap (P = ∞, pinv tronquée aux K premiers modes d'entraînement).

Dans F_K = vect{u_k·φ, k ≤ K}, fonctions propres généralisées g_j = v_j·φ : Σ_test v_j = μ_j Σ_train v_j, v_jᵀΣ_train v_j = 1
(‖g_j‖²_train = 1, ‖g_j‖²_test = μ_j). Pour toute référence f_ref ∈ F_K, ε = ω − f_ref :
    E_OOD(K) = E_test^min(K) + Σ_j (μ_j ⟨ε,g_j⟩_train − ⟨ε,g_j⟩_test)² / μ_j,
et avec f_ref = meilleur prédicteur sur la loi uniforme ρ_u = (2/3)ρ_train + (1/3)ρ_test :
    E_OOD(K) = E_test^min(K) + Σ_j (μ_j + 2)²/μ_j · e_j²,  e_j = ⟨ε_u, g_j⟩_train,  Σ_j e_j² ≤ ε²_{u,train}.
Vérifie l'identité sur les vrais spectres et imprime ε²_u(K), μ_max(K), E_OOD(K).
"""
import sys, os
import numpy as np
from scipy.linalg import eigh
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t4_spectres_reels import analyse


def runge_point(a, K):
    lam, S, c, ct, o2t, om2 = a["lam"], a["S"], a["c"], a["ct"], a["o2t"], a["omega2"]
    L = lam[:K]; St = S[:K, :K]; Str = np.diag(L)
    Su = 2 / 3 * Str + 1 / 3 * St; Cu = 2 / 3 * c[:K] + 1 / 3 * ct[:K]; o2u = 2 / 3 * om2 + 1 / 3 * o2t
    thu = np.linalg.solve(Su, Cu)
    eps_u2 = o2u - Cu @ thu                                   # ‖ε_u‖² sur ρ_u
    eps_tr2 = om2 - 2 * c[:K] @ thu + thu @ Str @ thu         # ‖ε_u‖² sur ρ_train
    mu, V = eigh(St, Str)
    a_ = V.T @ c[:K]; b_ = V.T @ ct[:K]
    # E_test^min(K) = ⟨ω²⟩_test − C_testᵀ Σ_test⁺ C_test (pinv : test_cov n'est précis qu'à ~1e-8 relatif,
    # certains μ_j sont ~0 ou légèrement < 0 ; la forme Σ b_j²/μ_j est numériquement instable)
    Emin = o2t - ct[:K] @ np.linalg.lstsq(St, ct[:K], rcond=1e-10)[0]
    W = c[:K] / L
    Eood = o2t - 2 * ct[:K] @ W + W @ St @ W
    e = a_ - V.T @ Str @ thu
    shift = np.sum((mu + 2) ** 2 / mu * e**2)
    return dict(K=K, lamK=L[-1] / lam[0], Eood=Eood, Emin=Emin, shift=shift, eps_u2=eps_u2, eps_tr2=eps_tr2,
                mu_max=mu.max(), n_mu1=int(np.sum(mu > 1)), mu=mu, e2=e**2)


def runge_curve(a, nK=16):
    lam = a["lam"]
    Kmax = int(np.sum(lam > 1e-13 * lam[0]))
    Ks = sorted(set(int(x) for x in np.round(np.geomspace(2, Kmax, nK))))
    return [runge_point(a, K) for K in Ks]


if __name__ == "__main__":
    for wtag in ["0.6283", "1.2566"]:
        for tpl in ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]:
            a = analyse(tpl, wtag, verbose=False)
            print(f"{tpl} ωmax={a['wmax']:.4f}")
            for r in runge_curve(a, nK=10):
                assert abs(r["Eood"] - r["Emin"] - r["shift"]) < 1e-3 * r["Eood"] + 1e-9, (r["K"], r["Eood"], r["Emin"], r["shift"])
                print(f"  K={r['K']:4d} λK/λ1={r['lamK']:.0e}  E_OOD={r['Eood']:.2e}  E_min={max(r['Emin'],0):.0e}  "
                      f"ε²_u={r['eps_u2']:.1e}  μ_max={r['mu_max']:.1e}  #μ>1={r['n_mu1']:3d}  "
                      f"borne (1+μ_max)ε²_u,tr={(1+r['mu_max'])*r['eps_tr2']:.1e}")
    print("identité vérifiée sur tous les points")
