"""T4 — Test de taille finie de theory.ood_error (données gaussiennes, régime sur-paramétré P = 0,24 N, λ_r petit).
Spectre fonction de k/N (densité spectrale fixe), 4 réalisations de (Σ_test, C_test) par N ; on compare la moyenne
empirique de E_OOD (300 jeux) à la théorie. Attendu : écart relatif → 0 sans biais systématique quand N croît."""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
def run(N, frac_P, lr0, NT, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(1, N + 1) * 250 / N
    lam = x ** -1.3
    t = 0.4 * x ** -1.2 * 250 / N
    c = np.sqrt(t * lam) * rng.choice([-1, 1], N)
    s2 = 0.03; om2 = t.sum() + s2; theta = c / lam
    amp = np.zeros(N); amp[int(0.08*N):int(0.24*N)] = 0.3
    B = np.diag(np.sqrt(lam)) + np.outer(amp, rng.standard_normal(N)) / np.sqrt(N) + 0.05 * rng.standard_normal((N, N)) * np.sqrt(lam)[None, :] * np.sqrt(250/N)
    Sig_t = B @ B.T
    theta_t = theta + 0.5 * amp * rng.standard_normal(N)* np.sqrt(250/N)
    C_t = Sig_t @ theta_t; om2_t = theta_t @ Sig_t @ theta_t + 0.05
    P = int(frac_P * N); lr = lr0 * P / 60
    v = []
    for _ in range(NT):
        X = rng.standard_normal((P, N)) * np.sqrt(lam)
        y = X @ theta + np.sqrt(s2) * rng.standard_normal(P)
        W = X.T @ np.linalg.solve(X @ X.T + lr * np.eye(P), y) if P < N else np.linalg.solve(X.T @ X + lr*np.eye(N), X.T @ y)
        v.append(om2_t - 2 * C_t @ W + W @ Sig_t @ W)
    E = th.ood_error(lam, c, om2, Sig_t, C_t, om2_t, P, lr)
    return np.mean(v), np.std(v)/np.sqrt(NT), E
for N in [125, 250, 500, 1000]:
    rel = []
    for seed in range(4):
        m, e, E = run(N, 0.24, 0.01, 300, 100*N + seed)
        rel.append(((m - E)/E, e/E))
    r = np.array(rel)
    w = 1/r[:,1]**2
    print(N, " ".join(f"{a:+.3f}±{b:.3f}" for a,b in r), f" | moyenne pondérée {np.sum(w*r[:,0])/w.sum():+.4f}±{1/np.sqrt(w.sum()):.4f}")
