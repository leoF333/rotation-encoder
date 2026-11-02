"""T3 — Variété 1D : distribution de E_gen (queues lourdes dues aux « trous » d'échantillonnage aléatoire)
et échantillonnage régulier (grille) vs aléatoire i.i.d.  Modèle : toy1d.py."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th
from toy1d import build, target_coeffs, feats, ridge, pop_error_uniform
rng = np.random.default_rng(7)
N = 300
for spec, a, target in [("pow", 2.0, "omega"), ("exp", 0.15, "omega"), ("exp", 0.15, "marche")]:
    k, lam = build(N, spec, a)
    c, y2, f = target_coeffs(k, lam, target)
    print(f"\n=== {spec} a={a} cible {target}")
    print(f"{'P':>4} {'lam_r':>6} | {'théorie':>9} | {'iid moy.':>9} {'iid méd.':>9} {'iid q90':>9} | {'grille':>9} | {'max trou':>8}")
    for P in [40, 80, 320]:
        for lr_rel in [1e-8, 1e-3]:
            lr = lr_rel * P * lam[0]
            pred = th.gen_error(lam, c, y2, P, lr)
            e, gaps = [], []
            for _ in range(200):
                w = np.sort(rng.uniform(0, 1, P))
                e.append(pop_error_uniform(ridge(feats(w, k, lam), f(w), lr), c, y2, lam))
                gaps.append(np.max(np.diff(np.r_[0, w, 1])))
            e = np.array(e)
            wg = (np.arange(P) + 0.5) / P
            eg = pop_error_uniform(ridge(feats(wg, k, lam), f(wg), lr), c, y2, lam)
            # corrélation log erreur / log plus grand trou
            rho = np.corrcoef(np.log(e), np.log(gaps))[0, 1]
            print(f"{P:4d} {lr_rel:6.0e} | {pred:9.2e} | {e.mean():9.2e} {np.median(e):9.2e} {np.quantile(e, .9):9.2e} |"
                  f" {eg:9.2e} | corr(log E, log trou)={rho:+.2f}")
