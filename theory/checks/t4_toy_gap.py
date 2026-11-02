"""T4 — Gap sur la variété 1D jouet : rôle de la LARGEUR DE BANDE des features.

Features φ_k(ω) = √(2 s_k) cos(π k ω), k = 0..N−1 (φ_0 = √s_0). Poids spectraux s_k :
  (A) « lisse »   : s_k = exp(−k²/(2B²))           (analogue du blob gaussien : coupure douce super-exponentielle)
  (B) « bords »   : s_k = (1+k)^−2 pour k ≤ k_max   (analogue des bords francs : loi de puissance + coupure pixel)
Entraînement ω ~ U([0,1/3] ∪ [2/3,1]), test ω ~ U[1/3,2/3] (gap), cible y = ω.
Moments population par quadrature de Gauss–Legendre ; base propre d'entraînement ; theory.ood_error ;
simulation directe (P tirages i.i.d. sur le créneau, ridge, E_OOD exact par forme quadratique).
Nombre de Slepian attendu (fonctions concentrées dans le gap) ≈ (bande angulaire max πk_max)·(longueur 1/3)/π = k_max/3.
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
rng = np.random.default_rng(31)

def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w

def feats(w, s):
    k = np.arange(s.size)
    F = np.sqrt(2 * s) * np.cos(np.pi * np.outer(w, k))
    F[:, 0] = np.sqrt(s[0])
    return F

def moments(nodes, weights, s):
    F = feats(nodes, s)
    Wt = weights / weights.sum()
    return (F * Wt[:, None]).T @ F, F.T @ (Wt * nodes), np.sum(Wt * nodes**2)

def setup(s):
    x1, w1 = gl(0, 1/3, 600); x2, w2 = gl(2/3, 1, 600)
    S_tr, C_tr, o2_tr = moments(np.r_[x1, x2], np.r_[w1, w2], s)
    xt, wt = gl(1/3, 2/3, 600)
    S_te, C_te, o2_te = moments(xt, wt, s)
    ev, U = np.linalg.eigh(S_tr)
    ev, U = ev[::-1], U[:, ::-1]
    return dict(S_tr=S_tr, C_tr=C_tr, o2_tr=o2_tr, S_te=S_te, C_te=C_te, o2_te=o2_te, ev=ev, U=U,
                c=U.T @ C_tr, tc=U.T @ S_te @ U, tcp=U.T @ C_te)

def simulate(s, d, P, lr, ntrial=40):
    v = []
    for _ in range(ntrial):
        w = np.where(rng.random(P) < 0.5, rng.uniform(0, 1/3, P), rng.uniform(2/3, 1, P))
        F = feats(w, s)
        W = F.T @ np.linalg.solve(F @ F.T + lr * np.eye(P), w)
        v.append(d["o2_te"] - 2 * d["C_te"] @ W + W @ d["S_te"] @ W)
    return np.mean(v), np.std(v) / np.sqrt(ntrial), np.median(v)

N = 160
k = np.arange(N)
P = 300
print("Var(ω) dans le gap = (1/3)²/12 =", round((1/3)**2/12, 5))
print(f"{'famille':>16} | {'#μ>1':>5} {'#μ>1e3':>6} | {'E_OOD pop. tronq.1e-4':>21} | {'théorie P=300':>13} | {'simulation (moy±err, méd)':>30}")
for fam, pars in [("lisse B", [2, 4, 8, 16]), ("bords k_max", [6, 12, 24, 48])]:
    for p in pars:
        s = np.exp(-k**2 / (2 * p**2)) if fam == "lisse B" else np.where(k <= p, (1.0 + k) ** -2, 0.0)
        d = setup(s)
        keep = d["ev"] > 1e-13 * d["ev"][0]
        mu = th.generalized_ratio_spectrum(d["ev"][keep], d["tc"][np.ix_(keep, keep)])
        Et = th.ood_error_truncated(d["ev"], d["c"], d["tc"], d["tcp"], d["o2_te"], tau=1e-4)
        lr = 1e-4 * d["ev"][0] * P
        Eth = th.ood_error(d["ev"][keep], d["c"][keep], d["o2_tr"], d["tc"][np.ix_(keep, keep)], d["tcp"][keep], d["o2_te"], P, lr)
        m, e, med = simulate(s, d, P, lr)
        print(f"{fam:>12} {p:3d} | {np.sum(mu > 1):5d} {np.sum(mu > 1e3):6d} | {Et:21.3e} | {Eth:13.3e} | {m:10.3e}±{e:.1e} (méd {med:.2e})")
