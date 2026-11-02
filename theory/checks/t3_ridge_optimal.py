"""T3 — Ridge optimal.
(1) professeur « isotrope » t_k = τ² λ_k : λ_r* = σ²/τ² exactement, pour tout P (prédiction bayésienne).
(2) cas général : la condition de stationnarité  κ(1−γ) Σ t_k λ_k/(λ_k+κ)³ = (σ²+B0)(1/P) Σ λ_k²/(λ_k+κ)³
    est satisfaite à l'optimum numérique.
(3) simulation gaussienne : l'optimum de la courbe simulée E_gen(λ_r) coïncide avec λ_r* théorique.
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
rng = np.random.default_rng(3)
N = 300; k = np.arange(1, N + 1); lam = k ** -1.4
tau2, s2 = 0.7, 0.05
t = tau2 * lam; c = np.sqrt(t * lam)
for P in [50, 300, 1000]:
    lr, E = th.optimal_ridge(lam, c, t.sum() + s2, P)
    print(f"(1) P={P:5d}: λ_r* numérique = {lr:.5f}   σ²/τ² = {s2/tau2:.5f}")
# (2)
t2 = k ** -2.5; c2 = np.sqrt(t2 * lam)
for P in [50, 300]:
    lr, E = th.optimal_ridge(lam, c2, t2.sum() + s2, P)
    op = th.order_parameters(lam, c2, t2.sum() + s2, P, lr)
    kap, gam = op["kappa"], op["gamma"]
    lhs = kap * (1 - gam) * np.sum(t2 * lam / (lam + kap) ** 3)
    rhs = (s2 + op["B0"]) * np.sum(lam**2 / (lam + kap) ** 3) / P
    print(f"(2) P={P}: λ_r*={lr:.4g}, κ*={kap:.4g}, condition lhs/rhs = {lhs/rhs:.5f}")
# (3)
P = 100
lrs = np.logspace(-3, 1, 9)
theta = c2 / lam
sim = []
for lr in lrs:
    v = []
    for _ in range(40):
        X = rng.standard_normal((P, N)) * np.sqrt(lam)
        y = X @ theta + np.sqrt(s2) * rng.standard_normal(P)
        W = X.T @ np.linalg.solve(X @ X.T + lr * np.eye(P), y)
        v.append(s2 + np.sum(lam * (W - theta) ** 2))
    sim.append(np.mean(v))
thy = [th.gen_error(lam, c2, t2.sum() + s2, P, lr) for lr in lrs]
print("(3) λ_r  :", np.round(lrs, 4))
print("    sim  :", np.round(sim, 4))
print("    théo :", np.round(thy, 4), " λ_r* théo =", round(th.optimal_ridge(lam, c2, t2.sum() + s2, P)[0], 4))
