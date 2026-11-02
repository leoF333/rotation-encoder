"""B3 — quelle architecture d'autoencodeur pour extraire ω et reconstruire la vidéo ?

Protocole (template arc par défaut, comme A4) : ω ∈ [0.05, 2π/T] ; entraînement sur le créneau
[0.05, a] ∪ [b, 2π/T] (a, b = tiers de l'intervalle), P = 4000 vidéos ; tests : 1000 vidéos du créneau
(« en distribution ») et 500 vidéos du gap [a, b]. Adam (lr 1e-3, cosinus), batch 128, nombre de pas fixé
(6400 pas = 200 époques à P = 4000), MSE par pixel, MPS float32, seeds 0, 1, 2.

Métriques : MSE de reconstruction (distribution / gap), qualité du latent sur tout l'intervalle (distribution ∪ gap) :
Spearman |ρ|(z, ω) (k = 1), R² de ω prédit par une fonction monotone de z (isotone ; k = 2 : angle polaire),
R² 5-NN (injectivité, toute dimension), R² monotone restreint au gap ; coût : paramètres, temps.

Configurations : voir CONFIGS (nom -> encodeur caché, décodeur caché, k, activation, P).
Un json par (template, config, seed) dans results/b3/ (idempotent). Usage :
  python scripts/b3_architectures.py [--template arc] [--configs all|nom1,nom2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from aerot import render, ae
from aerot.render import get_device
from aerot.paths import RES

D, T = 28, 10
N = D * D * T
WLO, WHI = 0.05, 2 * np.pi / T
GA, GB = WLO + (WHI - WLO) / 3, WLO + 2 * (WHI - WLO) / 3
STEPS = 6400
OUT = RES / "b3"

# nom : (encodeur caché, décodeur caché, k, activation, P)
CONFIGS = {
    "lin-shallow":    ((), (), 1, "relu", 4000),
    "lin-mid":        ((), (256,), 1, "relu", 4000),
    "lin-deep":       ((), (64, 256, 1024), 1, "relu", 4000),
    "lin-deepwide":   ((), (128, 512, 2048), 1, "relu", 4000),
    "lin-deep-silu":  ((), (64, 256, 1024), 1, "silu", 4000),
    "mlp-mid":        ((256,), (256,), 1, "relu", 4000),
    "mlp-deep":       ((256,), (64, 256, 1024), 1, "relu", 4000),
    "deep-deep":      ((256, 256), (64, 256, 1024), 1, "relu", 4000),
    "lin-shallow-k2": ((), (), 2, "relu", 4000),
    "lin-deep-k2":    ((), (64, 256, 1024), 2, "relu", 4000),
    "mlp-deep-k2":    ((256,), (64, 256, 1024), 2, "relu", 4000),
    "lin-deep-P500":  ((), (64, 256, 1024), 1, "relu", 500),
    "lin-deep-P1000": ((), (64, 256, 1024), 1, "relu", 1000),
    "mlp-deep-P500":  ((256,), (64, 256, 1024), 1, "relu", 500),
    "mlp-deep-P1000": ((256,), (64, 256, 1024), 1, "relu", 1000),
}


def sample_creneau(rng, n):
    L1, L2 = GA - WLO, WHI - GB
    k = rng.random(n) < L1 / (L1 + L2)
    u = rng.random(n)
    return np.where(k, WLO + u * L1, GB + u * L2)


def datasets(template, dev):
    r = np.random.default_rng(0)
    om_tr = sample_creneau(r, 4000)
    om_in = np.sort(sample_creneau(np.random.default_rng(1), 1000))
    om_gap = np.sort(np.random.default_rng(2).uniform(GA, GB, 500))
    mk = lambda o: render.make_videos(o, template, d=D, T=T, device=dev, dtype=torch.float32)  # noqa: E731
    return dict(tr=(om_tr, mk(om_tr)), ind=(om_in, mk(om_in)), gap=(om_gap, mk(om_gap)))


def run(template, name, seed, data, dev):
    f = OUT / f"{template}_{name}_s{seed}.json"
    if f.exists():
        return json.loads(f.read_text())
    enc, dec, k, act, P = CONFIGS[name]
    torch.manual_seed(seed)
    if "shared" in name:
        model = ae.SharedFrameAE(N, T, D, enc, dec, act)
    else:
        model = ae.AE(N, name, k, enc, dec, act)
    om_tr, Xtr = data["tr"]
    Xtr, om_tr = Xtr[:P], om_tr[:P]
    epochs = int(round(STEPS / int(np.ceil(P / 128))))
    info = ae.train(model, Xtr, epochs=epochs, batch=128, lr=1e-3, device=dev, seed=seed, log_every=max(1, epochs // 10))
    res = dict(template=template, config=name, enc_hidden=enc, dec_hidden=dec, k=k, act=act, P=P, seed=seed,
               epochs=epochs, n_params=ae.n_params(model), train_time_s=info["train_time_s"], history=info["history"])
    zs = {}
    for part in ["ind", "gap"]:
        om, X = data[part]
        z, e = ae.encode_eval(model, X, device=dev)
        zs[part] = (z, om)
        res[f"mse_{part}"] = float(e.mean())
        res[f"mse_{part}_max"] = float(e.max())
    z_all = np.concatenate([zs["ind"][0], zs["gap"][0]])
    om_all = np.concatenate([zs["ind"][1], zs["gap"][1]])
    o = np.argsort(om_all)
    z_all, om_all = z_all[o], om_all[o]
    met = ae.latent_metrics(z_all if k > 1 else z_all.reshape(-1), om_all)
    res.update({f"{kk}_all": v for kk, v in met.items()})
    mg = ae.latent_metrics(zs["gap"][0] if k > 1 else zs["gap"][0].reshape(-1), zs["gap"][1])
    res.update({f"{kk}_gap": v for kk, v in mg.items()})
    res["collapsed"] = bool(np.std(z_all, axis=0).max() < 1e-6)
    np.savez(OUT / f"{template}_{name}_s{seed}.npz", z_all=z_all, om_all=om_all)
    f.write_text(json.dumps(res, indent=1, default=float))
    print(f.name, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in res.items()
                   if kk.startswith(("mse_ind", "mse_gap", "R2_monotone", "R2_knn", "spearman", "train_time"))}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="arc")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    data = datasets(a.template, dev)
    names = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for n in names:
            t0 = time.time()
            run(a.template, n, seed, data, dev)


if __name__ == "__main__":
    main()
