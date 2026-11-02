"""Spectres du second moment par l'astuce de Gram (float64, CPU).

Pour des échantillons pondérés (phi_i, w_i), sum w_i = 1, on pose A = diag(sqrt(w)) Phi (n x N).
Sigma = A^T A (N x N) et G = A A^T (n x n) ont les mêmes valeurs propres non nulles ; si
G v_k = lambda_k v_k alors u_k = A^T v_k / sqrt(lambda_k) est vecteur propre unitaire de Sigma.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def _eigh_desc(G):
    lam, V = sla.eigh(G, driver="evd")
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]
    lam = np.clip(lam, 0.0, None)
    return lam, V


class GramSpectrum:
    """Décomposition spectrale de Sigma = <phi phi^T> (non centré) via la matrice de Gram.

    Attributs : lam (valeurs propres décroissantes), V (vecteurs propres de G), A, sw = sqrt(w),
    omegas, c_proj (u_k . C), phi_mean_proj (u_k . phibar), omega2, omega_mean.
    """

    def __init__(self, Phi: np.ndarray, w: np.ndarray, omegas: np.ndarray, centered: bool = False):
        Phi = np.asarray(Phi, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        self.sw = np.sqrt(w)
        self.w = w
        self.omegas = np.asarray(omegas, dtype=np.float64)
        self.A = self.sw[:, None] * Phi
        G = self.A @ self.A.T
        self.centered = centered
        if centered:
            Pm = np.eye(len(w)) - np.outer(self.sw, self.sw)
            G = Pm @ G @ Pm
            self.Pm = Pm
        self.lam, self.V = _eigh_desc(G)
        self.omega_mean = float(w @ self.omegas)
        self.omega2 = float(w @ self.omegas**2)
        sq = np.sqrt(self.lam)
        if centered:
            self.c_proj = sq * (self.V.T @ (self.sw * (self.omegas - self.omega_mean)))
            self.phi_mean_proj = np.zeros_like(self.lam)
        else:
            self.c_proj = sq * (self.V.T @ (self.sw * self.omegas))
            self.phi_mean_proj = sq * (self.V.T @ self.sw)

    @property
    def N(self):
        return self.A.shape[1]

    def n_above(self, rel: float) -> int:
        """Nombre K de modes avec lambda_k / lambda_1 > rel."""
        return int(np.sum(self.lam > rel * self.lam[0]))

    def eigvecs(self, K: int) -> np.ndarray:
        """Vecteurs propres u_1..u_K de Sigma (N x K)."""
        lam = self.lam[:K]
        Amat = self.A if not self.centered else self.Pm @ self.A
        return (Amat.T @ self.V[:, :K]) / np.sqrt(lam)[None, :]

    def weights_trunc(self, rel: float = 1e-4) -> np.ndarray:
        """Régression pseudo-inverse tronquée : W = sum_{lambda_k > rel lambda_1} c_k/lambda_k u_k."""
        K = self.n_above(rel)
        U = self.eigvecs(K)
        return U @ (self.c_proj[:K] / self.lam[:K])

    def weights_ridge(self, lam_r: float, K: int | None = None, rel_floor: float = 1e-15) -> np.ndarray:
        """Ridge sur le second moment : W = (Sigma + lam_r I)^{-1} C = sum_k c_k/(lambda_k+lam_r) u_k.

        Ici lam_r est la régularisation « par échantillon » (Hamiltonien normalisé par P) :
        pour H = 1/2 sum_mu (...)^2 + lambda_r/2 |W|^2 on a lam_r = lambda_r / P."""
        if K is None:
            K = self.n_above(rel_floor)
        U = self.eigvecs(K)
        return U @ (self.c_proj[:K] / (self.lam[:K] + lam_r))

    def energy_fraction(self, K: int) -> float:
        return float(self.lam[:K].sum() / self.lam.sum())

    def test_quantities(self, Phi_t: np.ndarray, w_t: np.ndarray, omegas_t: np.ndarray, K: int):
        """Matrice de second moment test dans la base propre d'entraînement (K premiers modes),
        projections u_k . C_test, <omega^2>_test, u_k . phibar_test, trace de Sigma_test."""
        w_t = np.asarray(w_t, dtype=np.float64)
        w_t = w_t / w_t.sum()
        B = np.sqrt(w_t)[:, None] * np.asarray(Phi_t, dtype=np.float64)
        U = self.eigvecs(K)
        Y = B @ U  # (n_t, K)
        cov = Y.T @ Y
        c_t = Y.T @ (np.sqrt(w_t) * omegas_t)
        m_t = Y.T @ np.sqrt(w_t)
        return dict(test_cov_in_train_basis=cov, test_c_proj=c_t, test_phi_mean_proj=m_t,
                    test_omega2=float(w_t @ omegas_t**2), test_omega_mean=float(w_t @ omegas_t),
                    test_trace=float((B * B).sum()))


def pca_projection(Phi: np.ndarray, n_comp: int = 2):
    """PCA centrée classique (échantillons équipondérés) : renvoie les projections (n x n_comp),
    la variance expliquée relative et les composantes."""
    X = np.asarray(Phi, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    lam, V = _eigh_desc(G)
    proj = V[:, :n_comp] * np.sqrt(lam[:n_comp])[None, :]
    comps = (Xc.T @ V[:, :n_comp]) / np.sqrt(lam[:n_comp])[None, :]
    return proj, lam[:n_comp] / lam.sum(), comps
