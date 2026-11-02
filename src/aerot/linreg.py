"""Régression linéaire (encodeur W sans biais) à P échantillons : chemin ridge exact via une seule
décomposition spectrale, et évaluation des erreurs sur des ensembles pondérés (quadratures « population »).

Conventions (docs/CONVENTIONS.md) : H(W) = ½ Σ_μ (ω_μ − W·φ_μ)² + (λ_r/2)‖W‖²  =>  W = (ΦᵀΦ + λ_r I)⁻¹ Φᵀω.
Erreurs par échantillon, sans ½, sans terme ridge.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


class RidgePath:
    """Pré-calcule la décomposition de ΦΦᵀ (P ≤ N, forme duale) ou de ΦᵀΦ (P > N, forme primale)
    pour obtenir W(λ_r) pour de nombreux λ_r, et les erreurs sur des ensembles d'évaluation.

    `evals` : dict nom -> (X_e (n_e x N), ω_e (n_e), poids w_e (somme 1)).
    """

    def __init__(self, Phi: np.ndarray, om: np.ndarray, evals: dict, primal: bool | None = None):
        Phi = np.asarray(Phi, np.float64)
        self.P, self.N = Phi.shape
        self.om = np.asarray(om, np.float64)
        self.primal = (self.P > self.N) if primal is None else primal
        if not self.primal:
            G = Phi @ Phi.T
            D, V = sla.eigh(G, driver="evd")
            self.D = np.clip(D, 0, None)
            self.V = V
            self.a = V.T @ self.om                     # V^T ω
            self.KV = {k: (X @ Phi.T) @ V for k, (X, _, _) in evals.items()}
        else:
            S = Phi.T @ Phi
            D, U = sla.eigh(S, driver="evd")
            self.D = np.clip(D, 0, None)
            self.U = U
            self.a = U.T @ (Phi.T @ self.om)           # U^T Φ^T ω
            self.XU = {k: X @ U for k, (X, _, _) in evals.items()}
            self.om2 = float(self.om @ self.om)
        self.evals = {k: (np.asarray(o, np.float64), np.asarray(w, np.float64) / np.sum(w))
                      for k, (_, o, w) in evals.items()}

    def _coef(self, lam_r=None, rel_cut=None):
        Dmax = self.D.max()
        if rel_cut is not None:  # pseudo-inverse tronquée (seuil relatif sur les valeurs propres de ΦΦᵀ/P)
            keep = self.D > rel_cut * Dmax
            inv = np.where(keep, 1.0 / np.where(keep, self.D, 1.0), 0.0)
        else:
            inv = 1.0 / (self.D + lam_r)
        return inv

    def errors(self, lam_r=None, rel_cut=None):
        inv = self._coef(lam_r, rel_cut)
        b = self.a * inv
        out = {}
        if not self.primal:
            pred_tr = self.V @ (self.D * b)
            out["Etrain"] = float(np.mean((self.om - pred_tr) ** 2))
            out["normW2"] = float(np.sum(self.D * b**2))
            for k, KV in self.KV.items():
                o, w = self.evals[k]
                out[k] = float(w @ (o - KV @ b) ** 2)
        else:
            # W = U b ; E_train = (|ω|² − 2 Wᵀ Φᵀω + Wᵀ ΦᵀΦ W)/P
            out["Etrain"] = float((self.om2 - 2 * b @ self.a + np.sum(self.D * b**2)) / self.P)
            out["normW2"] = float(np.sum(b**2))
            for k, XU in self.XU.items():
                o, w = self.evals[k]
                out[k] = float(w @ (o - XU @ b) ** 2)
        return out


def gaussian_surrogate(Xpop: np.ndarray, om_pop: np.ndarray, w_pop: np.ndarray, P: int,
                       rng: np.random.Generator, chunk: int = 2000):
    """Échantillons gaussiens (φ, ω) ayant exactement les mêmes moyennes et covariances (donc les mêmes
    seconds moments non centrés Σ, C, ⟨ω²⟩) que la loi « population » discrète {(φ_i, ω_i), w_i}."""
    w = np.asarray(w_pop, np.float64) / np.sum(w_pop)
    m_phi = w @ Xpop
    m_om = float(w @ om_pop)
    sw = np.sqrt(w)
    A = sw[:, None] * (Xpop - m_phi)
    a = sw * (om_pop - m_om)
    Phi = np.empty((P, Xpop.shape[1]))
    om = np.empty(P)
    for s in range(0, P, chunk):
        g = rng.standard_normal((min(chunk, P - s), len(w)))
        Phi[s:s + chunk] = m_phi + g @ A
        om[s:s + chunk] = m_om + g @ a
    return Phi, om
