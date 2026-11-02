"""Modèle continu (sans pixels) : le template n'intervient que par son spectre de puissance angulaire.

Pour un template s0(r, θ) = Σ_k a_k(r) e^{ikθ} observé dans le disque r ≤ r_max, le produit scalaire de deux
vidéos vaut
    φ(ω)·φ(ω') = Σ_t C((ω − ω') t),   C(Δ) = ∫ s0(x) s0(R_Δ x) dx = 2π [E_0 + 2 Σ_{k≥1} E_k cos(kΔ)],
avec E_k = ∫ |a_k(r)|² r dr. Tout le problème de régression (spectre de Σ, projections, erreurs dans le gap)
ne dépend donc du template qu'à travers la suite (E_k). On le réalise par des « features » exactes
    [√(2πE_0)] ⊕ [√(4πE_k) cos(k t ω), √(4πE_k) sin(k t ω)]_{k≥1, t=0..T-1}
dont le noyau linéaire est exactement le noyau ci-dessus (pas d'erreur d'interpolation).
"""
from __future__ import annotations

import numpy as np
import torch

from .templates import evaluate, template_params


def angular_power(name, params=None, d=28, r_max=None, nr=800, nth=4096, oversample=4):
    """Spectre de puissance angulaire E_k (k = 0..nth/2) du template continu, intégré sur r ≤ r_max
    (défaut : disque inscrit dans l'image, r_max = d/2). Intégration radiale au point milieu, avec
    sur-échantillonnage angulaire/radial pour les templates discontinus."""
    p = template_params(name, d) if params is None else params
    r_max = d / 2 if r_max is None else r_max
    rs = (np.arange(nr * oversample) + 0.5) * r_max / (nr * oversample)
    th = torch.arange(nth, dtype=torch.float64) * (2 * np.pi / nth)
    Ek = np.zeros(nth // 2 + 1)
    dr = rs[1] - rs[0]
    for s in range(0, len(rs), 256):
        R = torch.as_tensor(rs[s:s + 256])[:, None]
        vals = evaluate(name, R * torch.cos(th)[None, :], R * torch.sin(th)[None, :], **p).numpy()
        ak = np.fft.rfft(vals, axis=1) / nth
        Ek += (np.abs(ak) ** 2 * rs[s:s + 256, None]).sum(0) * dr
    return Ek


def features(omegas, Ek, T=10, kmax=None, tol=1e-16):
    """Features exactes du modèle continu (n x F)."""
    Ek = np.asarray(Ek, float)
    if kmax is None:
        good = np.nonzero(Ek > tol * Ek.max())[0]
        kmax = int(good.max())
    k = np.arange(1, kmax + 1)
    om = np.asarray(omegas, float)
    cols = [np.full((len(om), 1), np.sqrt(2 * np.pi * Ek[0] * T))]  # T features constantes regroupées
    amp = np.sqrt(4 * np.pi * Ek[1:kmax + 1])
    for t in range(1, T):
        ph = np.outer(om, k * t)
        cols.append(np.cos(ph) * amp)
        cols.append(np.sin(ph) * amp)
    # t = 0 : cos(0) = 1 pour tout k -> contribue Σ_k 4π E_k à la constante
    cols[0] = np.full((len(om), 1), np.sqrt(2 * np.pi * (Ek[0] + 2 * Ek[1:kmax + 1].sum()) * 1
                                            + 2 * np.pi * Ek[0] * (T - 1)))
    return np.hstack(cols)


def kernel_row_check(Ek, T, om1, om2, kmax):
    """C-sommation directe pour vérification : Σ_t C((ω1−ω2)t)."""
    k = np.arange(1, kmax + 1)
    s = 0.0
    for t in range(T):
        s += 2 * np.pi * (Ek[0] + 2 * np.sum(Ek[1:kmax + 1] * np.cos(k * (om1 - om2) * t)))
    return s


def effective_bandwidth(Ek, q=0.999):
    """Plus petit k tel que 2Σ_{1≤j≤k}E_j + E_0 ≥ q × (E_0 + 2Σ_{j≥1}E_j) (bande contenant une fraction q de l'énergie)."""
    tot = Ek[0] + 2 * Ek[1:].sum()
    cum = Ek[0] + 2 * np.cumsum(Ek[1:])
    return int(np.searchsorted(cum / tot, q) + 1)
