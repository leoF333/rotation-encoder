"""T2 — Entropie avec ridge : S/N = 1/2(1+ln(2pi/beta)) - (1/2N) E ln det(X^T X + lam_r I),
avec l'équivalent déterministe (issu du col RS) :
  E ln det(X^T X + lam_r I) ~ sum_k ln(lam_r + P lam_k/(1+chi)) + P ln(1+chi) - P chi/(1+chi).
Vérifie aussi : pour alpha<1, S ~ -(N-P)/2 ln lam_r -> +inf quand lam_r -> 0 (directions plates).
"""
import numpy as np
from scipy.optimize import brentq
rng = np.random.default_rng(3)

def chi_of(lam, P, lam_r):
    return np.exp(brentq(lambda lc: np.exp(lc) - np.sum(lam/(lam*P/(1+np.exp(lc)) + lam_r)), -40, 40))

N = 300; k = np.arange(1, N+1); lam = 5*k**-1.5
for P in [120, 600]:
    for lam_r in [1e-4, 1e-2, 1.0]:
        chi = chi_of(lam, P, lam_r)
        pred = np.sum(np.log(lam_r + P*lam/(1+chi))) + P*np.log1p(chi) - P*chi/(1+chi)
        def lndet():
            X = rng.standard_normal((P, N))*np.sqrt(lam)
            return np.linalg.slogdet(X.T @ X + lam_r*np.eye(N))[1]
        sims = [lndet() for _ in range(10)]
        print(f"P={P:4d} lam_r={lam_r:7.0e}  (1/N)E lndet sim={np.mean(sims)/N:9.4f}  pred={pred/N:9.4f}   chi={chi:.3e}")
