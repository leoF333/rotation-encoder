"""B3 — références pour les MSE du balayage d'architectures (mêmes jeux de données que b3_architectures.py).

Pour chaque template : MSE (par pixel) du prédicteur constant (vidéo moyenne d'entraînement) et de la
reconstruction PCA (meilleure reconstruction linéaire affine de rang k, k = 1, 2, 5, 10), en distribution et
dans le gap. Sortie : results/b3_baselines.json + results/b3_baselines.md.
Usage : python scripts/b3_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_architectures import datasets  # noqa: E402
from aerot.paths import RES
from aerot.render import get_device


def main():
    dev = get_device()
    out, lines = {}, ["# B3 — MSE de référence (prédicteur constant, PCA affine de rang k ; entraînement P = 4000 sur le créneau)", "",
                      "| template | test | constant | PCA k=1 | PCA k=2 | PCA k=5 | PCA k=10 |", "|---|---|---|---|---|---|---|"]
    for tpl in ["arc", "gaussian", "cross"]:
        data = datasets(tpl, dev)
        X = data["tr"][1].cpu().numpy().astype(np.float64)
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        out[tpl] = {}
        for part in ["ind", "gap"]:
            Y = data[part][1].cpu().numpy().astype(np.float64) - mu
            r = {"constant": float((Y ** 2).mean())}
            for k in [1, 2, 5, 10]:
                V = Vt[:k]
                r[f"pca{k}"] = float(((Y - (Y @ V.T) @ V) ** 2).mean())
            out[tpl][part] = r
            lines.append(f"| {tpl} | {part} | " + " | ".join(f"{r[c]:.2e}" for c in ["constant", "pca1", "pca2", "pca5", "pca10"]) + " |")
        print(tpl, out[tpl], flush=True)
    (RES / "b3_baselines.json").write_text(json.dumps(out, indent=1))
    (RES / "b3_baselines.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
