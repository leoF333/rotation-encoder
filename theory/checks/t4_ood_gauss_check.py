"""T4 — Vérif. de theory.ood_error sur données gaussiennes (entraînement gaussien Σ_train, test = autre loi
de seconds moments Σ_test, C_test, ⟨ω²⟩_test arbitraires, NON diagonaux dans la base d'entraînement).
E_OOD(W) exact = test_omega2 − 2 C_test·W + Wᵀ Σ_test W pour chaque W simulé ; moyenne sur jeux."""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
rng = np.random.default_rng(21)
NTRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N = 250
k = np.arange(1, N + 1)
lam = k ** -1.3
t = 0.4 * k ** -1.2
c = np.sqrt(t * lam) * rng.choice([-1, 1], N)
s2 = 0.03
om2 = t.sum() + s2
theta = c / lam
# loi test : Σ_test = B Bᵀ avec B mélangeant et amplifiant les modes 20..60 (« gap-concentrés »)
B = np.diag(np.sqrt(lam)) + 0.0
amp = np.zeros(N); amp[20:60] = 0.3
B = B + np.outer(amp, rng.standard_normal(N)) / np.sqrt(N) + 0.05 * rng.standard_normal((N, N)) * np.sqrt(lam)[None, :]
Sig_t = B @ B.T
theta_t = theta + 0.5 * amp * rng.standard_normal(N)          # la cible test n'est pas θ*·φ : mauvaise spécification
C_t = Sig_t @ theta_t
om2_t = theta_t @ Sig_t @ theta_t + 0.05
print(f"{'P':>5} {'lam_r':>6} | {'E_OOD sim':>18} | {'théorie':>9} (biais, variance)")
for P, lr in [(60, 0.01), (60, 1.0), (250, 0.0), (400, 0.3), (1000, 0.0), (1000, 3.0)]:
    v = []
    for _ in range(NTRIAL):
        X = rng.standard_normal((P, N)) * np.sqrt(lam)
        y = X @ theta + np.sqrt(s2) * rng.standard_normal(P)
        if lr == 0 and P <= N:
            W = np.linalg.pinv(X) @ y
        elif P >= N:
            W = np.linalg.solve(X.T @ X + lr * np.eye(N), X.T @ y)
        else:
            W = X.T @ np.linalg.solve(X @ X.T + lr * np.eye(P), y)
        v.append(om2_t - 2 * C_t @ W + W @ Sig_t @ W)
    if P == N and lr == 0:
        print(f"{P:5d} {lr:6.2f} | (pic d'interpolation, sauté)"); continue
    r = th.ood_error(lam, c, om2, Sig_t, C_t, om2_t, P, lr, return_parts=True)
    print(f"{P:5d} {lr:6.2f} | {np.mean(v):9.4f}±{np.std(v)/np.sqrt(len(v)):.4f} | {r['Eood']:9.4f} ({r['bias']:.4f}, {r['variance']:.4f})")
