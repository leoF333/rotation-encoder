"""Templates continus s0(x, y).

Coordonnées : x = colonne - c, y = ligne - c, en pixels, avec c = (d-1)/2 le centre de rotation
(convention « tableau » : l'axe y pointe vers le bas à l'affichage). Chaque template est une
fonction vectorisée (torch) de (x, y) renvoyant l'intensité dans [0, 1].

Paramètres par défaut (d = 28) estimés en numérisant les figures du mémoire (cf. docs/REPORT_numerics.md,
section « Décisions ») ; pour d != 28 les longueurs sont mises à l'échelle d/28.
"""
from __future__ import annotations

import math
import torch

# Paramètres par défaut pour d = 28 (longueurs en pixels).
DEFAULTS = {
    # tache gaussienne : sigma, distance au centre r0, angle initial theta0 (Fig. 2a : ~45°)
    "gaussian": dict(sigma=3.0, r0=8.0, theta0=math.pi / 4),
    # dipôle : deux gaussiennes diamétralement opposées
    "dipole": dict(sigma=3.0, r0=8.0, theta0=math.pi / 4),
    # quart de disque de rayon R, coin au centre de rotation (Fig. 7)
    "quarter_circle": dict(R=10.0),
    # croix centrée : bras de largeur w, demi-longueur L (Fig. 2b : w = 4, bras jusqu'au bord)
    "cross": dict(w=4.0, L=14.0),
    # disque centré (témoin négatif)
    "disk": dict(R=8.0),
    # segment (« aiguille ») partant du centre, longueur L, épaisseur w (Fig. 6)
    "segment": dict(L=10.0, w=1.0),
    # arc de cercle : rayon R, épaisseur w, ouverture dtheta (Fig. 3)
    "arc": dict(R=11.0, w=1.0, theta0=0.0, dtheta=math.pi / 2),
    # --- templates de contrôle (hors mémoire) ---
    # un seul bras de la croix (même largeur, même longueur) : témoin sans symétrie d'ordre 4
    "arm": dict(w=4.0, L=14.0),
}

LENGTH_KEYS = {"sigma", "r0", "R", "w", "L"}

# Templates dont le profil d'intensité est discontinu (bords francs)
DISCONTINUOUS = {"quarter_circle", "cross", "disk", "segment", "arc", "arm"}

LABELS_FR = {
    "gaussian": "gaussienne",
    "dipole": "dipôle",
    "quarter_circle": "quart de cercle",
    "cross": "croix",
    "disk": "disque (témoin)",
    "segment": "segment",
    "arc": "arc de cercle",
    "arm": "bras seul",
}


def template_params(name: str, d: int = 28, **overrides) -> dict:
    """Paramètres du template `name` pour une image d x d (mise à l'échelle d/28 des longueurs)."""
    p = dict(DEFAULTS[name])
    s = d / 28.0
    for k in p:
        if k in LENGTH_KEYS:
            p[k] = p[k] * s
    p.update(overrides)
    return p


def _gauss(x, y, x0, y0, sigma):
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def evaluate(name: str, x: torch.Tensor, y: torch.Tensor, **p) -> torch.Tensor:
    """Intensité du template `name` aux points (x, y) (tenseurs de même forme)."""
    one = torch.ones((), dtype=x.dtype, device=x.device)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    if name == "gaussian":
        x0 = p["r0"] * math.cos(p["theta0"])
        y0 = p["r0"] * math.sin(p["theta0"])
        return _gauss(x, y, x0, y0, p["sigma"])
    if name == "dipole":
        x0 = p["r0"] * math.cos(p["theta0"])
        y0 = p["r0"] * math.sin(p["theta0"])
        return _gauss(x, y, x0, y0, p["sigma"]) + _gauss(x, y, -x0, -y0, p["sigma"])
    if name == "quarter_circle":
        m = (x >= 0) & (y <= 0) & (x**2 + y**2 <= p["R"] ** 2)
        return torch.where(m, one, zero)
    if name == "cross":
        h = p["w"] / 2
        inside = (x.abs() <= p["L"]) & (y.abs() <= p["L"])
        m = inside & ((x.abs() < h) | (y.abs() < h))
        return torch.where(m, one, zero)
    if name == "disk":
        return torch.where(x**2 + y**2 <= p["R"] ** 2, one, zero)
    if name == "segment":
        m = (x >= 0) & (x <= p["L"]) & (y >= 0) & (y <= p["w"])
        return torch.where(m, one, zero)
    if name == "arm":
        m = (x >= 0) & (x <= p["L"]) & (y.abs() < p["w"] / 2)
        return torch.where(m, one, zero)
    if name == "arc":
        r = torch.sqrt(x**2 + y**2)
        th = torch.remainder(torch.atan2(y, x) - p["theta0"], 2 * math.pi)
        m = (r >= p["R"] - p["w"] / 2) & (r <= p["R"] + p["w"] / 2) & (th <= p["dtheta"])
        return torch.where(m, one, zero)
    raise ValueError(f"template inconnu : {name}")


# templates du mémoire (exportés par défaut)
TEMPLATE_NAMES = ["gaussian", "dipole", "quarter_circle", "cross", "disk", "segment", "arc"]
