"""V2b : sensibilité du pic de E(ω) dans le gap (2π/T) aux détails numériques, templates discontinus."""
import numpy as np, json, functools
import vlib
wmax = 2 * np.pi / vlib.T
om_eval = np.linspace(0, wmax, 1201)
gap = (om_eval > wmax / 3) & (om_eval < 2 * wmax / 3)
variants = {
    "cross base (n=4000, ss=4)": (vlib.cross, 4000, 4),
    "cross n=8000": (vlib.cross, 8000, 4),
    "cross ss=8": (vlib.cross, 4000, 8),
    "cross demi-longueur 13.5": (functools.partial(vlib.cross, half_len=13.5), 4000, 4),
    "segment base (n=4000, ss=4)": (vlib.segment, 4000, 4),
    "segment n=8000": (vlib.segment, 8000, 4),
    "segment ss=8": (vlib.segment, 4000, 8),
    "segment centré sur y=0 (±0.5)": (lambda x, y: ((x >= 0) & (x <= 10) & (np.abs(y) <= 0.5)).astype(float), 4000, 4),
}
res = {}
for lab, (f, n, ss) in variants.items():
    pop = vlib.population_svd(f, "creneau", wmax, n=n, ss=ss)
    Phi = vlib.render(f, om_eval, ss=ss)
    row = {}
    for cut in [1e-4, 1e-8]:
        W, K = vlib.truncated_W(pop, cut)
        row[f"{cut:g}"] = dict(K=K, peak=float(((om_eval - Phi @ W) ** 2)[gap].max()))
    res[lab] = row
    print(f"{lab:32s} 1e-4: K={row['0.0001']['K']:4d} pic={row['0.0001']['peak']:.2e} | 1e-8: K={row['1e-08']['K']:4d} pic={row['1e-08']['peak']:.2e}", flush=True)
    json.dump(res, open("results/v2b_sensibilite.json", "w"), indent=1)
