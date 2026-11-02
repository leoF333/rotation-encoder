"""Autoencodeurs (PyTorch) et métriques de l'espace latent.

Architectures (N = T d^2, k = dimension latente) :
- ``sym3``  : x -> z = A x + a (linéaire) -> x^ = sigmoid(B z + b)            (« 3 couches » : entrée, latent, sortie)
- ``sym5``  : x -> act(h) -> z -> act(h) -> sigmoid(N)                        (« 5 couches »)
- ``asym``  : z = A x (+a) linéaire -> décodeur profond [k -> h1 -> ... -> N]  (encodeur shallow, décodeur deep)
- ``mlpenc``: encodeur à 1 couche cachée (N -> h -> k) + décodeur profond      (B3)
- ``deepenc``: encodeur profond (N -> h -> h -> k) + décodeur profond           (B3)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr

from .render import get_device

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def mlp(sizes, act="relu", last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ACTS[act]())
    if last_act == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self, N, arch="asym", k=1, enc_hidden=(), dec_hidden=(64, 256, 1024), act="relu",
                 out="sigmoid"):
        super().__init__()
        self.arch, self.k = arch, k
        self.encoder = mlp([N, *enc_hidden, k], act)
        self.decoder = mlp([k, *dec_hidden, N], act, last_act=out)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SharedFrameAE(nn.Module):
    """Encodeur linéaire x -> z ∈ R, décodeur « physique » partagé entre les images : image t = g(z·t).

    g : R -> R^{d²} (MLP, sortie sigmoïde) est commun aux T images ; il encode le biais inductif
    φ_t(ω) = R_{ωt}·template (l'image ne dépend que de l'angle ωt). Sortie au format (n, T·d²), images consécutives.
    """

    def __init__(self, N, T, d, enc_hidden=(), dec_hidden=(64, 256, 1024), act="relu", out="sigmoid"):
        super().__init__()
        self.arch, self.k, self.T = "shared", 1, T
        self.encoder = mlp([N, *enc_hidden, 1], act)
        self.decoder = mlp([1, *dec_hidden, d * d], act, last_act=out)
        self.register_buffer("t", torch.arange(T, dtype=torch.float32).view(1, T, 1))

    def forward(self, x):
        z = self.encoder(x)                               # (n, 1)
        frames = self.decoder((z.unsqueeze(1) * self.t))  # (n, T, d²)
        return frames.reshape(x.shape[0], -1), z


def build(arch, N, k=1, width=128, dec_hidden=(64, 256, 1024), act="relu"):
    if arch == "sym3":
        return AE(N, arch, k, (), (), act)
    if arch == "sym5":
        return AE(N, arch, k, (width,), (width,), act)
    if arch == "asym":
        return AE(N, arch, k, (), tuple(dec_hidden), act)
    if arch == "mlpenc":
        return AE(N, arch, k, (width,), tuple(dec_hidden), act)
    if arch == "deepenc":
        return AE(N, arch, k, (width, width), tuple(dec_hidden), act)
    raise ValueError(arch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, Xtr, Xte=None, epochs=300, batch=128, lr=1e-3, wd=0.0, device=None, seed=0,
          log_every=0, max_minutes=10.0):
    """Adam + décroissance cosinus, perte MSE (moyenne par pixel). Renvoie l'historique."""
    device = get_device() if device is None else device
    torch.manual_seed(seed)
    model.to(device)
    Xtr = Xtr.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = math.ceil(len(Xtr) / batch)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        for i in range(steps_per_epoch):
            xb = Xtr[perm[i * batch:(i + 1) * batch]]
            xh, _ = model(xb)
            loss = ((xh - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            hist.append((ep, float(loss.detach().cpu())))
        if (time.time() - t0) / 60 > max_minutes:
            hist.append(("stopped_time", ep))
            break
    return dict(history=hist, train_time_s=time.time() - t0, epochs_done=ep + 1)


@torch.no_grad()
def encode_eval(model, X, device=None, batch=1024):
    device = get_device() if device is None else device
    model.eval()
    zs, errs = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        xh, z = model(xb)
        zs.append(z.cpu())
        errs.append(((xh - xb) ** 2).mean(1).cpu())
    model.train()
    return torch.cat(zs).numpy().astype(np.float64), torch.cat(errs).numpy().astype(np.float64)


@torch.no_grad()
def reconstruct(model, X, device=None):
    device = get_device() if device is None else device
    model.eval()
    xh, z = model(X.to(device))
    model.train()
    return xh.cpu().numpy(), z.cpu().numpy()


# ------------------------------------------------------------------ métriques du latent

def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))


def monotone_r2(z, omega):
    """R^2 de la meilleure fonction monotone omega = f(z) (régression isotone, croissante ou décroissante)."""
    o = np.argsort(z)
    y = omega[o]
    best = -np.inf
    for inc in (True, False):
        fit = isotonic_regression(y, increasing=inc).x
        best = max(best, r2(y, fit))
    return best


def latent_metrics(z, omega):
    """Pour z de dimension 1 : Spearman |rho|, R^2 monotone (isotone), R^2 lin/sin de z vs omega (comme la Fig. 4).
    Pour k > 1 : R^2 d'un k-NN (k=5) de omega sur z (injectivité) et R^2 monotone de la meilleure
    projection angulaire (angle polaire de (z1, z2) centrés) si k = 2."""
    z = np.asarray(z, float)
    out = {}
    if z.ndim == 1 or z.shape[1] == 1:
        z1 = z.reshape(-1)
        out["spearman_abs"] = float(abs(spearmanr(z1, omega)[0]))
        out["R2_monotone"] = monotone_r2(z1, omega)
        A = np.vstack([omega, np.ones_like(omega)]).T
        coef, *_ = np.linalg.lstsq(A, z1, rcond=None)
        out["R2_lin_z_vs_omega"] = r2(z1, A @ coef)
    else:
        zc = z - z.mean(0)
        ang = np.arctan2(zc[:, 1], zc[:, 0])
        best = -np.inf
        # angle défini à une coupure près : on essaie plusieurs coupures
        for cut in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            a = np.mod(ang - cut, 2 * np.pi)
            best = max(best, monotone_r2(a, omega))
        out["R2_monotone_angle"] = best
        out["spearman_abs"] = float("nan")
    out["R2_knn"] = knn_r2(z.reshape(len(omega), -1), omega)
    return out


def knn_r2(z, omega, k=5):
    """R^2 de omega prédit par la moyenne de ses k plus proches voisins dans l'espace latent (leave-one-out)."""
    z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    D = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    pred = omega[idx].mean(1)
    return r2(omega, pred)
