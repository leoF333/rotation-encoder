"""Génération vectorisée des vidéos phi(omega) (CPU float64 ou GPU MPS float32).

Deux méthodes de rendu :

- ``analytic`` : l'image t est la fonction continue du template évaluée aux coordonnées
  tournées, phi_t(p) = s0(R(-omega t) p), moyennée sur ss x ss sous-pixels (ss = 1 :
  échantillonnage ponctuel au centre des pixels ; ss > 1 : anti-crénelage « boîte »).
- ``interp`` : le template est d'abord rastérisé à t = 0 (même règle, sur-échantillonnage ss),
  puis l'image est tournée par interpolation bilinéaire (zéro hors de l'image), comme le ferait
  ``torch.nn.functional.grid_sample`` / ``scipy.ndimage.rotate(order=1)``.

Convention : rotation autour de c = (d-1)/2, l'image t = 0, ..., T-1 est tournée de omega * t.
Le vecteur phi est la vidéo aplatie dans l'ordre (t, ligne, colonne) : N = T d^2.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .templates import evaluate, template_params


# Rendu par défaut (cf. docs/REPORT_numerics.md, §1.1) : analytique, sur-échantillonnage 4x4 (anti-crénelage boîte).
DEFAULT_METHOD = "analytic"
DEFAULT_SS = 4


def get_device(prefer: str = "auto") -> torch.device:
    """Best available device: Apple MPS, then CUDA, then CPU (or the one requested)."""
    if prefer not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError(prefer)
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pixel_grid(d: int, ss: int, dtype, device):
    """Coordonnées (X, Y) des sous-pixels, forme (d, d, ss*ss) ; X = colonne - c, Y = ligne - c."""
    c = (d - 1) / 2
    idx = torch.arange(d, dtype=dtype, device=device) - c
    sub = (torch.arange(ss, dtype=dtype, device=device) + 0.5) / ss - 0.5
    coord = idx[:, None] + sub[None, :]  # (d, ss)
    Y = coord[:, None, :, None].expand(d, d, ss, ss)  # ligne i, sous-ligne a
    X = coord[None, :, None, :].expand(d, d, ss, ss)  # colonne j, sous-colonne b
    return X.reshape(d, d, ss * ss), Y.reshape(d, d, ss * ss)


def raster(name: str, d: int = 28, ss: int = 1, params: dict | None = None,
           dtype=torch.float64, device="cpu", angle: float = 0.0) -> torch.Tensor:
    """Image d x d du template tourné de `angle` (rendu analytique)."""
    p = template_params(name, d) if params is None else params
    X, Y = _pixel_grid(d, ss, dtype, device)
    ca, sa = math.cos(angle), math.sin(angle)
    Xr = X * ca + Y * sa
    Yr = -X * sa + Y * ca
    return evaluate(name, Xr, Yr, **p).mean(-1)


def _bilinear(img: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Interpolation bilinéaire de img (d x d) aux indices flottants (px = colonne, py = ligne),
    valeur nulle hors de l'image."""
    d = img.shape[0]
    pad = 2
    ip = torch.zeros((d + 2 * pad, d + 2 * pad), dtype=img.dtype, device=img.device)
    ip[pad:pad + d, pad:pad + d] = img
    x = px + pad
    y = py + pad
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    wx = x - x0
    wy = y - y0
    x0 = x0.long().clamp(0, d + 2 * pad - 2)
    y0 = y0.long().clamp(0, d + 2 * pad - 2)
    W = d + 2 * pad
    flat = ip.reshape(-1)
    i00 = y0 * W + x0
    v00 = flat[i00]
    v01 = flat[i00 + 1]
    v10 = flat[i00 + W]
    v11 = flat[i00 + W + 1]
    return (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy)
            + v10 * (1 - wx) * wy + v11 * wx * wy)


def make_videos(omegas, name: str, d: int = 28, T: int = 10, method: str = DEFAULT_METHOD,
                ss: int = DEFAULT_SS, params: dict | None = None, device="cpu",
                dtype=torch.float64, chunk: int | None = None, flat: bool = True) -> torch.Tensor:
    """Vidéos phi(omega) pour un tableau d'omegas.

    Renvoie un tenseur (n, N) si flat, sinon (n, T, d, d), sur `device` et en `dtype`.
    """
    p = template_params(name, d) if params is None else params
    om = torch.as_tensor(np.asarray(omegas, dtype=np.float64)).to(dtype=dtype, device=device)
    n = om.shape[0]
    t = torch.arange(T, dtype=dtype, device=device)
    c = (d - 1) / 2
    if method == "analytic":
        X, Y = _pixel_grid(d, ss, dtype, device)
        per = T * d * d * ss * ss
    elif method == "interp":
        img0 = raster(name, d, ss, p, dtype, device)
        X, Y = _pixel_grid(d, 1, dtype, device)
        X, Y = X[..., 0], Y[..., 0]
        per = T * d * d
    else:
        raise ValueError(method)
    if chunk is None:
        chunk = max(1, int(6e6 // per))
    out = torch.empty((n, T, d, d), dtype=dtype, device=device)
    for s in range(0, n, chunk):
        a = om[s:s + chunk, None] * t[None, :]  # (m, T)
        ca, sa = torch.cos(a), torch.sin(a)
        if method == "analytic":
            ca = ca[:, :, None, None, None]
            sa = sa[:, :, None, None, None]
            Xr = X * ca + Y * sa
            Yr = -X * sa + Y * ca
            out[s:s + chunk] = evaluate(name, Xr, Yr, **p).mean(-1)
        else:
            ca = ca[:, :, None, None]
            sa = sa[:, :, None, None]
            Xr = X * ca + Y * sa
            Yr = -X * sa + Y * ca
            out[s:s + chunk] = _bilinear(img0, Xr + c, Yr + c)
    return out.reshape(n, -1) if flat else out


# ---------------------------------------------------------------- distributions de omega

def interval_list(dist: str, wmax: float):
    """Intervalles (support uniforme) de la distribution `dist`."""
    if dist == "uniform":
        return [(0.0, wmax)]
    if dist == "creneau":
        return [(0.0, wmax / 3), (2 * wmax / 3, wmax)]
    if dist == "gap":
        return [(wmax / 3, 2 * wmax / 3)]
    raise ValueError(dist)


def sample_omegas(dist: str, wmax: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tirage i.i.d. de n vitesses selon `dist` (uniforme sur la réunion des intervalles)."""
    ivs = interval_list(dist, wmax)
    lens = np.array([b - a for a, b in ivs])
    k = rng.choice(len(ivs), size=n, p=lens / lens.sum())
    u = rng.random(n)
    a = np.array([iv[0] for iv in ivs])[k]
    return a + u * lens[k]


def quadrature(dist: str, wmax: float, n: int, q: int = 8):
    """Quadrature de Gauss-Legendre composite (q noeuds par panneau), ~n noeuds au total,
    poids normalisés à 1 : <f>_rho ≈ sum_i w_i f(omega_i)."""
    ivs = interval_list(dist, wmax)
    lens = np.array([b - a for a, b in ivs])
    xg, wg = np.polynomial.legendre.leggauss(q)
    nodes, weights = [], []
    for (a, b), L in zip(ivs, lens):
        m = max(1, int(round(n * L / lens.sum() / q)))
        edges = np.linspace(a, b, m + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            h = (hi - lo) / 2
            nodes.append(lo + h * (xg + 1))
            weights.append(h * wg)
    nodes = np.concatenate(nodes)
    weights = np.concatenate(weights)
    return nodes, weights / weights.sum()
