"""Tests de src/aerot/theory.py (exécuter : python theory/checks/test_theory.py ; ~1 min CPU).

Chaque test compare une formule à une simulation directe sur données gaussiennes synthétiques
(moyenne sur des jeux d'entraînement indépendants) ou à une identité exacte.
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

rng = np.random.default_rng(10)
FAIL = []

def check(name, sim, sim_err, pred, rel=0.03, atol=1e-12):
    ok = abs(sim - pred) <= max(4 * sim_err, rel * abs(pred), atol)
    print(f"[{'OK ' if ok else 'ECHEC'}] {name}: sim={sim:.5g}±{sim_err:.2g} théorie={pred:.5g}")
    if not ok:
        FAIL.append(name)

def simulate(lam, c, omega2, P, lam_r, ntrial=40, mean=None):
    """Données gaussiennes centrées dans la base propre ; renvoie (E_train, E_gen) moyens ± err."""
    N = lam.size
    theta = c / lam
    s2 = omega2 - np.sum(c**2 / lam)
    et, eg = [], []
    for _ in range(ntrial):
        X = rng.standard_normal((P, N)) * np.sqrt(lam)
        y = X @ theta + np.sqrt(s2) * rng.standard_normal(P)
        if P >= N:
            W = np.linalg.solve(X.T @ X + lam_r * np.eye(N), X.T @ y)
        else:  # forme duale (plus rapide, et lam_r=0 -> norme minimale)
            W = X.T @ (np.linalg.pinv(X @ X.T + lam_r * np.eye(P)) @ y) if lam_r == 0 else \
                X.T @ np.linalg.solve(X @ X.T + lam_r * np.eye(P), y)
        et.append(np.mean((y - X @ W) ** 2))
        eg.append(s2 + np.sum(lam * (W - theta) ** 2))
    f = lambda v: (np.mean(v), np.std(v) / np.sqrt(len(v)))
    return f(et), f(eg)

# spectre en loi de puissance, professeur t_k ~ k^-b
N = 400
k = np.arange(1, N + 1)
lam = 2.0 * k ** -1.3
t = 0.5 * k ** -1.1
c = np.sqrt(t * lam) * rng.choice([-1, 1], N)
omega2 = t.sum() + 0.05

for P, lam_r in [(100, 1e-3), (100, 1.0), (400, 0.3), (800, 0.0), (800, 5.0), (200, 0.0)]:
    (et, ete), (eg, ege) = simulate(lam, c, omega2, P, lam_r)
    op = th.order_parameters(lam, c, omega2, P, lam_r)
    check(f"E_gen   P={P} lam_r={lam_r}", eg, ege, op["Egen"])
    check(f"E_train P={P} lam_r={lam_r}", et, ete, op["Etrain"])
    assert abs(op["bias_modes"].sum() + op["var_modes"].sum() - (op["Egen"] - op["sigma2"])) < 1e-9 * op["Egen"]

# mémoire corrigé (covariance centrée + moyenne) == order_parameters (moments non centrés), ridgeless alpha>1
Nm, Pm = 50, 150
Um = np.linalg.qr(rng.standard_normal((Nm, Nm)))[0]
Sc = (Um * np.linspace(0.5, 2, Nm)) @ Um.T
phibar = rng.standard_normal(Nm) * 0.4
thm = rng.standard_normal(Nm) * 0.3
wbar, s2m = 0.8, 0.2
Cc = Sc @ thm; varw = thm @ Sc @ thm + s2m
Si = np.linalg.inv(Sc)
J = dict(Jcc=Cc @ Si @ Cc, Jphic=phibar @ Si @ Cc, Jphiphi=phibar @ Si @ phibar)
Sig = Sc + np.outer(phibar, phibar); Cu = Cc + wbar * phibar
ev, U = np.linalg.eigh(Sig)
op = th.order_parameters(ev, U.T @ Cu, varw + wbar**2, Pm, 0.0)
mm = th.memoire_ridgeless(**J, wbar=wbar, sigma2=varw, alpha=Pm / Nm)
mm_orig = th.memoire_ridgeless(**J, wbar=wbar, sigma2=varw, alpha=Pm / Nm, corrected=False)
check("mémoire corrigé E_gen = order_parameters (ridgeless)", mm["Egen"], 0.0, op["Egen"], rel=1e-8)
check("mémoire corrigé E_train = order_parameters (ridgeless)", mm["Etrain"], 0.0, op["Etrain"], rel=1e-8)
print(f"      (formule originale du mémoire, erreur E2 : E_train={mm_orig['Etrain']:.5g})")

# énergie libre RS vs exact
N2, P2, lr2, beta = 150, 90, 0.5, 2.0
lam2 = lam[:N2]; c2 = c[:N2]; om22 = np.sum(c2**2 / lam2) + 0.05
theta2 = c2 / lam2
lz = []
for _ in range(40):
    X = rng.standard_normal((P2, N2)) * np.sqrt(lam2)
    y = X @ theta2 + np.sqrt(0.05) * rng.standard_normal(P2)
    A = X.T @ X + lr2 * np.eye(N2)
    W = np.linalg.solve(A, X.T @ y)
    Hmin = 0.5 * np.sum((y - X @ W) ** 2) + 0.5 * lr2 * W @ W
    lz.append(-beta * Hmin + 0.5 * N2 * np.log(2 * np.pi / beta) - 0.5 * np.linalg.slogdet(A)[1])
fe = th.free_energy_rs(lam2, c2, om22, P2, lr2, beta)
check("<ln Z> RS", np.mean(lz), np.std(lz) / np.sqrt(40), fe["lnZ"], rel=0.01)

# --- T4 : hors distribution (loi de test gaussienne quelconque, non diagonale, cible mal spécifiée) ---
N3 = 200
lam3 = lam[:N3]; c3 = c[:N3]; om23 = np.sum(c3**2 / lam3) + 0.05; th3 = c3 / lam3
B3 = np.diag(np.sqrt(lam3)) + 0.05 * rng.standard_normal((N3, N3)) * np.sqrt(lam3)[None, :]
amp = np.zeros(N3); amp[10:40] = 0.3
B3 = B3 + np.outer(amp, rng.standard_normal(N3)) / np.sqrt(N3)
St3 = B3 @ B3.T
tht = th3 + 0.3 * amp * rng.standard_normal(N3)
Ct3 = St3 @ tht; o2t3 = tht @ St3 @ tht + 0.02
for P, lam_r in [(80, 0.5), (500, 2.0)]:
    v = []
    for _ in range(200):
        X = rng.standard_normal((P, N3)) * np.sqrt(lam3)
        y = X @ th3 + np.sqrt(0.05) * rng.standard_normal(P)
        W = X.T @ np.linalg.solve(X @ X.T + lam_r * np.eye(P), y) if P < N3 else np.linalg.solve(X.T @ X + lam_r * np.eye(N3), X.T @ y)
        v.append(o2t3 - 2 * Ct3 @ W + W @ St3 @ W)
    check(f"E_OOD   P={P} lam_r={lam_r}", np.mean(v), np.std(v) / np.sqrt(len(v)),
          th.ood_error(lam3, c3, om23, St3, Ct3, o2t3, P, lam_r), rel=0.03)
# en distribution, ood_error == gen_error
check("E_OOD(test = train) = E_gen", th.ood_error(lam3, c3, om23, np.diag(lam3), c3, om23, 300, 1.0), 0.0,
      th.gen_error(lam3, c3, om23, 300, 1.0), rel=1e-10)
# identité de Runge (exacte) et pinv tronquée
rd = th.runge_decomposition(lam3, c3, om23, St3, Ct3, o2t3, K=60)
check("Runge : E_OOD(K) = E_min + décalage", rd["Emin"] + rd["shift"], 0.0, rd["Eood"], rel=1e-8)
check("Runge : E_OOD(K) = ood_error_truncated", th.ood_error_truncated(lam3, c3, St3, Ct3, o2t3, tau=0.999 * lam3[59] / lam3[0]),
      0.0, rd["Eood"], rel=1e-10)

print("\nRÉSULTAT :", "tous les tests passent" if not FAIL else f"échecs : {FAIL}")
