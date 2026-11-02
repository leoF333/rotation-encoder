"""Ajustements simples (R², linéaire, sinusoïdal)."""
import numpy as np
from scipy.optimize import curve_fit


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))


def fit_lin(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


def fit_sin(x, y, bmax=60.0):
    """a sin(b x + c) + e : balayage fin de b (moindres carrés linéaires en (a cos c, a sin c, e)),
    puis raffinement non linéaire de tous les paramètres. Renvoie (R², valeurs ajustées, paramètres)."""
    f = lambda x, a, b, c, e: a * np.sin(b * x + c) + e  # noqa: E731
    if not np.all(np.isfinite(y)) or np.std(y) == 0:
        return float("nan"), np.full_like(y, np.mean(y)), [0.0, 0.0, 0.0, float(np.mean(y))]
    best_s, best_b, best_coef = -np.inf, None, None
    for b in np.linspace(0.5, bmax, 6000):
        A = np.vstack([np.sin(b * x), np.cos(b * x), np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        s = r2(y, A @ coef)
        if s > best_s:
            best_s, best_b, best_coef = s, b, coef
    p0 = [np.hypot(best_coef[0], best_coef[1]), best_b, np.arctan2(best_coef[1], best_coef[0]), best_coef[2]]
    try:
        p, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
        if r2(y, f(x, *p)) < best_s:
            p = p0
    except RuntimeError:
        p = p0
    return r2(y, f(x, *p)), f(x, *p), p
