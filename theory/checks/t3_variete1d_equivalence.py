"""T3 — Test de l'équivalence gaussienne pour des données sur une VARIÉTÉ 1D.

Modèle jouet : ω ~ U[0,1] ; features φ_0 = √λ_0 (constante), φ_k(ω) = √(2λ_k) cos(π k ω), k=1..N-1.
Base orthogonale sous ρ : Σ = diag(λ_k) exactement ; c_k = ⟨ω φ_k⟩ connus analytiquement :
  c_0 = √λ_0/2 ; c_k = √(2λ_k)((-1)^k − 1)/(π² k²)  (t_k = c_k²/λ_k = 8/(π⁴ k⁴) pour k impair).
Cible y = ω (déterministe : aucune « vraie » source de bruit ; σ²_res = queue Σ_{k≥N} t_k ≈ 0).
Deuxième cible (bords francs dans la cible) : y = 1[ω > 0.37] (t_k ~ k^-2).
E_gen exact par W : ⟨y²⟩ − 2 c·W + Σ λ_k W_k² (pas de Monte-Carlo sur le test).
Compare : théorie (src/aerot/theory.py), simulation sur la variété 1D, simulation gaussienne de même (λ, c).
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(5)

from toy1d import build, target_coeffs, feats, ridge

N = 300
for spec, a in [("pow", 2.0), ("exp", 0.15)]:
    k, lam = build(N, spec, a)
    for target in ["omega", "marche"]:
        c, y2, f = target_coeffs(k, lam, target)
        print(f"\n=== spectre {spec} a={a}, cible {target} : sigma2_res(N)={th.residual_variance(lam, c, y2):.2e}")
        print(f"{'P':>5} {'lam_r':>8} | {'théorie':>10} | {'variété 1D':>18} | {'gaussien':>18}")
        for P in [20, 80, 320]:
            for lr_rel in [1e-6, 1e-3]:
                lr = lr_rel * P * lam[0]
                pred = th.gen_error(lam, c, y2, P, lr)
                e1, e2 = [], []
                for _ in range(40):
                    w = rng.uniform(0, 1, P)
                    W = ridge(feats(w, k, lam), f(w), lr)
                    e1.append(y2 - 2 * c @ W + lam @ W**2)
                    theta = c / lam
                    s2 = max(y2 - np.sum(c**2 / lam), 0)
                    X = rng.standard_normal((P, N)) * np.sqrt(lam)
                    yy = X @ theta + np.sqrt(s2) * rng.standard_normal(P)
                    W = ridge(X, yy, lr)
                    e2.append(y2 - 2 * c @ W + lam @ W**2)
                m = lambda v: f"{np.mean(v):.3e}±{np.std(v)/np.sqrt(len(v)):.1e}"
                print(f"{P:5d} {lr_rel:8.0e} | {pred:10.3e} | {m(e1)} | {m(e2)}")
