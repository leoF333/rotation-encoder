"""B1 — validation des répliques : E_train, E_gen (en distribution), E_OOD (créneau -> gap) en fonction de P
(α = P/N) et du ridge λ_r, moyennés sur plusieurs tirages du jeu d'entraînement ; données de rotation vs
substituts gaussiens de mêmes moments ; effets de taille finie (d).

Parties (idempotentes, un json par configuration dans results/b1/) :
  --part main   : 4 templates x {uniform, creneau} x {real, surrogate} x P ∈ P_GRID x seeds (forme duale)
  --part large  : P > N (forme primale), gaussienne et croix, uniforme, real + surrogate
  --part finite : d ∈ {14, 20, 28} à α fixé, gaussienne et croix, uniforme
  --part stratified : ω tirés de façon stratifiée (gaussienne, quart de cercle), test du mécanisme des trous d'échantillonnage
  --part labelnoise : bruit d'étiquette σ_ε = 1e-3, P ∈ [30, 12000] (test de E_train = (1 − 1/α)σ²_res du mémoire)
Régresseurs : ridge λ_r = r·P·λ_1 (r = 1e-2, 1e-4, 1e-6, 1e-8, i.e. ridge « par échantillon » r·λ_1) ;
pinv tronquée (seuil relatif 1e-4 comme le mémoire, et 1e-10 ≈ norme minimale).
ω_max = 4π/T. Évaluation « population » par quadrature de Gauss-Legendre (2000 nœuds).
Usage : python scripts/b1_replicas.py --part main
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aerot import render, spectra
from aerot.linreg import RidgePath, gaussian_surrogate
from aerot.paths import RES

OUT = RES / "b1"
T = 10
WMAX = 4 * np.pi / T
RIDGE_R = [1e-2, 1e-4, 1e-6, 1e-8]
CUTS = [1e-4, 1e-10]
P_GRID = [5, 10, 20, 40, 80, 160, 320, 1000, 3000, 7000]
TEMPLATES = ["gaussian", "quarter_circle", "cross", "segment"]


class Pop:
    """Loi « population » (quadrature) pour un template/une distribution, + ensembles d'évaluation."""

    def __init__(self, name, dist, d=28, n_pop=4000, n_eval=2000):
        self.name, self.dist, self.d = name, dist, d
        om, w = render.quadrature(dist, WMAX, n_pop)
        self.X = render.make_videos(om, name, d=d, T=T).numpy()
        self.om, self.w = om, w
        S = spectra.GramSpectrum(self.X, w, om)
        self.lam1 = float(S.lam[0])
        self.evals = {}
        oe, we = render.quadrature(dist, WMAX, n_eval)
        self.evals["gen"] = (render.make_videos(oe, name, d=d, T=T).numpy(), oe, we)
        if dist == "creneau":
            og, wg = render.quadrature("gap", WMAX, n_eval)
            self.evals["ood"] = (render.make_videos(og, name, d=d, T=T).numpy(), og, wg)


def one(pop: Pop, kind, P, seed, primal=None, label_noise=0.0):
    tag = f"_noise{label_noise:g}" if label_noise > 0 else ""
    f = OUT / f"{pop.name}_{pop.dist}_d{pop.d}_{kind}_P{P}_s{seed}{tag}.json"
    if f.exists():
        return
    t0 = time.time()
    rng = np.random.default_rng(10_000 * seed + P + (7 if kind == "surrogate" else 0))
    if kind == "real":
        om = render.sample_omegas(pop.dist, WMAX, P, rng)
        Phi = render.make_videos(om, pop.name, d=pop.d, T=T).numpy()
    elif kind == "strat":   # tirage stratifié : un ω uniforme dans chacune des P cellules de même masse (pas de « trou »)
        ivs = render.interval_list(pop.dist, WMAX)
        lens = np.array([b - a for a, b in ivs])
        u = (np.arange(P) + rng.random(P)) / P * lens.sum()
        cum = np.concatenate([[0], np.cumsum(lens)])
        k = np.searchsorted(cum, u, side="right") - 1
        om = np.array([ivs[i][0] for i in k]) + (u - cum[k])
        Phi = render.make_videos(om, pop.name, d=pop.d, T=T).numpy()
    else:
        Phi, om = gaussian_surrogate(pop.X, pop.om, pop.w, P, rng)
    if label_noise > 0:   # bruit sur les étiquettes d'entraînement (évaluation sur ω exact)
        om = om + label_noise * rng.standard_normal(P)
    rp = RidgePath(Phi, om, pop.evals, primal=primal)
    res = dict(template=pop.name, dist=pop.dist, d=pop.d, N=Phi.shape[1], kind=kind, P=P, seed=seed,
               wmax=WMAX, lam1_pop=pop.lam1, label_noise=label_noise, results={})
    for r in RIDGE_R:
        res["results"][f"ridge{r:.0e}"] = dict(lam_r=r * P * pop.lam1, **rp.errors(lam_r=r * P * pop.lam1))
    for c in CUTS:
        res["results"][f"pinv{c:.0e}"] = rp.errors(rel_cut=c)
    res["time_s"] = time.time() - t0
    f.write_text(json.dumps(res))
    print(f.name, f"{res['time_s']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main")
    ap.add_argument("--templates", default=",".join(TEMPLATES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "main":
        for name in a.templates.split(","):
            for dist in ["uniform", "creneau"]:
                pop = Pop(name, dist)
                for seed in range(5):
                    for P in P_GRID:
                        if P >= 7000 and seed >= 3:
                            continue
                        for kind in ["real", "surrogate"]:
                            one(pop, kind, P, seed)
    elif a.part == "large":
        for name in ["gaussian", "cross"]:
            pop = Pop(name, "uniform")
            for seed in range(3):
                for P in [12000, 20000]:
                    for kind in ["real", "surrogate"]:
                        one(pop, kind, P, seed, primal=True)
    elif a.part == "labelnoise":
        # test de la formule du mémoire E_train = (1 − 1/α) σ²_res : bruit d'étiquette σ_ε = 1e-3 (σ²_res ≈ σ_ε²)
        for name in ["gaussian", "cross"]:
            pop = Pop(name, "uniform")
            for seed in range(3):
                for P in [30, 100, 300, 1000, 3000, 7000, 12000]:
                    for kind in ["real", "surrogate"]:
                        one(pop, kind, P, seed, primal=(P > 7840), label_noise=1e-3)
    elif a.part == "stratified":
        # mécanisme de l'écart à l'équivalence gaussienne : tirage stratifié des ω (même loi marginale, sans trous)
        for name in ["gaussian", "quarter_circle"]:
            for dist in ["uniform", "creneau"]:
                pop = Pop(name, dist)
                for seed in range(5):
                    for P in [5, 10, 20, 40, 80, 160, 320, 1000]:
                        one(pop, "strat", P, seed)
    elif a.part == "finite":
        for name in ["gaussian", "cross"]:
            for d in [14, 20, 28]:
                pop = Pop(name, "uniform", d=d)
                N = d * d * T
                for alpha in [0.1, 0.3, 1.0]:
                    P = int(round(alpha * N))
                    for seed in range(3):
                        one(pop, "real", P, seed)


if __name__ == "__main__":
    main()
