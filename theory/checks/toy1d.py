"""Modèle jouet « variété 1D » (outil commun aux checks T3/T4).

ω ~ ρ sur [0,1] ; features φ_0 = √λ_0, φ_k(ω) = √(2λ_k) cos(π k ω), k = 1..N−1.
Sous ω ~ U[0,1] : Σ = diag(λ_k) exactement, et pour une cible y(ω) : c_k = √λ_k g_k avec g_k les
coefficients de y dans la base orthonormée {1, √2 cos(πkω)}.
"""
import numpy as np


def build(N, spec, a):
    k = np.arange(N)
    lam = np.exp(-a * k) if spec == "exp" else (1.0 + k) ** (-a)
    return k, lam


def target_coeffs(k, lam, target, w0=0.37):
    """Renvoie (c, ⟨y²⟩, f) pour target ∈ {'omega', 'marche'} (sous ω ~ U[0,1])."""
    km = np.maximum(k, 1)
    if target == "omega":
        g = np.where(k == 0, 0.5, np.sqrt(2) * ((-1.0) ** k - 1) / (np.pi**2 * km**2))
        return g * np.sqrt(lam), 1 / 3, (lambda w: w)
    g = np.where(k == 0, 1 - w0, -np.sqrt(2) * np.sin(np.pi * km * w0) / (np.pi * km))
    return g * np.sqrt(lam), 1 - w0, (lambda w: (w > w0).astype(float))


def feats(w, k, lam):
    F = np.sqrt(2 * lam) * np.cos(np.pi * np.outer(w, k))
    F[:, 0] = np.sqrt(lam[0])
    return F


def ridge(F, y, lr):
    P, N = F.shape
    if P >= N:
        return np.linalg.solve(F.T @ F + lr * np.eye(N), F.T @ y)
    return F.T @ np.linalg.solve(F @ F.T + lr * np.eye(P), y)


def pop_error_uniform(W, c, y2, lam):
    """E_gen exact sous ω ~ U[0,1] : ⟨y²⟩ − 2 c·W + Σ λ_k W_k²."""
    return y2 - 2 * c @ W + lam @ W**2
