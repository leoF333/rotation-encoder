"""Bibliothèque de vérification indépendante (écrite sans consulter le code du projet).

Conventions reprises de docs/CONVENTIONS.md :
- grille d×d, centre c = (d-1)/2, coordonnées x = j - c (colonnes), y = c - i (lignes, vers le haut) ;
- image t (t = 0..T-1) = template tourné de +ωt : le pixel p voit le point R(-ωt)p du template ;
- rendu « analytique » : moyenne de la fonction continue sur ss×ss sous-pixels.
"""
import numpy as np

D, T = 28, 10


def subpixel_coords(d=D, ss=4):
    c = (d - 1) / 2
    o = (np.arange(ss) + 0.5) / ss - 0.5
    j, i = np.meshgrid(np.arange(d), np.arange(d))          # j = colonne, i = ligne
    x = (j[..., None, None] + o[None, None, None, :]) - c    # (d, d, ss, ss)
    y = c - (i[..., None, None] + o[None, None, :, None])
    x, y = np.broadcast_arrays(x, y)
    return x.reshape(d * d, ss * ss), y.reshape(d * d, ss * ss)


# ---- templates : fonctions f(x, y) -> intensité, vectorisées -------------------------------
def gaussian(x, y, sigma=3.0, r0=8.0, ang=np.pi / 4):
    x0, y0 = r0 * np.cos(ang), r0 * np.sin(ang)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def cross(x, y, half_width=2.0, half_len=14.0):
    ax, ay = np.abs(x), np.abs(y)
    return (((ax < half_width) & (ay < half_len)) | ((ay < half_width) & (ax < half_len))).astype(float)


def quarter_disc(x, y, R=10.0):
    return ((x >= 0) & (y >= 0) & (x * x + y * y < R * R)).astype(float)


def segment(x, y, L=10.0):
    return ((x >= 0) & (x <= L) & (y >= -1.0) & (y <= 0.0)).astype(float)


def arc(x, y, R=11.0, thick=1.0, a0=-3 * np.pi / 4, a1=-np.pi / 4):
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    return ((np.abs(r - R) < thick / 2) & (a >= a0) & (a <= a1)).astype(float)


TEMPLATES = dict(gaussian=gaussian, cross=cross, quarter_circle=quarter_disc, segment=segment, arc=arc)


def render(f, omegas, d=D, T_=T, ss=4, chunk=64):
    """Renvoie Φ de forme (len(omegas), d*d*T) en float64 ; ligne = vidéo aplatie (t, pixel)."""
    x, y = subpixel_coords(d, ss)
    omegas = np.asarray(omegas, float)
    out = np.empty((len(omegas), T_, d * d))
    ts = np.arange(T_)
    for s in range(0, len(omegas), chunk):
        th = omegas[s:s + chunk, None] * ts[None, :]              # (b, T)
        c, sn = np.cos(th)[..., None, None], np.sin(th)[..., None, None]
        xr = c * x + sn * y                                         # R(-θ)p
        yr = -sn * x + c * y
        out[s:s + chunk] = f(xr, yr).mean(-1)
    return out.reshape(len(omegas), -1)


# ---- quadrature (moyennes « population ») --------------------------------------------------
def gl_nodes(intervals, n_total, per_panel=8):
    """Gauss-Legendre composite sur une union d'intervalles, poids normalisés (densité uniforme)."""
    lens = np.array([b - a for a, b in intervals])
    xs, ws = np.polynomial.legendre.leggauss(per_panel)
    nodes, weights = [], []
    for (a, b), L in zip(intervals, lens):
        npan = max(1, int(round(n_total * L / lens.sum() / per_panel)))
        edges = np.linspace(a, b, npan + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            nodes.append((hi - lo) / 2 * xs + (hi + lo) / 2)
            weights.append((hi - lo) / 2 * ws)
    w = np.concatenate(weights)
    return np.concatenate(nodes), w / w.sum()


def support(kind, wmax):
    if kind == "uniform":
        return [(0.0, wmax)]
    if kind == "creneau":
        return [(0.0, wmax / 3), (2 * wmax / 3, wmax)]
    if kind == "gap":
        return [(wmax / 3, 2 * wmax / 3)]
    raise ValueError(kind)


def population_svd(f, kind, wmax, n=4000, **kw):
    """A = diag(√w) Φ ; Σ = AᵀA = Σ_k s_k² u_k u_kᵀ. Renvoie s (valeurs singulières), U (N×r), b = Vᵀ(√w ω)."""
    om, w = gl_nodes(support(kind, wmax), n)
    A = np.sqrt(w)[:, None] * render(f, om, **kw)
    V, s, Ut = np.linalg.svd(A, full_matrices=False)
    b = V.T @ (np.sqrt(w) * om)          # u_k·C = s_k b_k
    return dict(s=s, U=Ut.T, b=b, omega2=float(w @ om ** 2), nodes=om, weights=w)


def truncated_W(pop, rel_eig_cut):
    """Pseudo-inverse tronquée : garde λ_k/λ_1 = (s_k/s_1)² > rel_eig_cut. W = Σ b_k/s_k u_k."""
    s = pop["s"]
    keep = (s / s[0]) ** 2 > rel_eig_cut
    return pop["U"][:, keep] @ (pop["b"][keep] / s[keep]), int(keep.sum())
