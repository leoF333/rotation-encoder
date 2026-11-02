"""V2 : expérience de gap à P = ∞ (moyennes population par quadrature), recalcul indépendant."""
import numpy as np, json, time, sys
import vlib

names = ["gaussian", "segment", "quarter_circle", "cross"]
res = {}
for wlabel, wmax in [("2pi/T", 2 * np.pi / vlib.T), ("4pi/T", 4 * np.pi / vlib.T)]:
    om_eval = np.linspace(0, wmax, 1201)
    for name in names:
        t0 = time.time()
        f = vlib.TEMPLATES[name]
        pop = vlib.population_svd(f, "creneau", wmax, n=4000)
        Phi_eval = vlib.render(f, om_eval)
        for cut in [1e-4, 1e-8]:
            W, K = vlib.truncated_W(pop, cut)
            E = (om_eval - Phi_eval @ W) ** 2
            g_t = (om_eval > wmax / 3) & (om_eval < 2 * wmax / 3)
            g_35 = (om_eval > 0.35 * wmax) & (om_eval < 0.65 * wmax)
            out = ~((om_eval >= wmax / 3) & (om_eval <= 2 * wmax / 3))
            r = dict(K=K, peak_gap_tiers=float(E[g_t].max()), peak_gap_035=float(E[g_35].max()),
                     max_out=float(E[out].max()), mean_gap_tiers=float(E[g_t].mean()))
            res[f"{wlabel}|{name}|{cut:g}"] = r
            print(f"{wlabel:6s} {name:15s} cut={cut:g}  K={K:4d}  pic gap(tiers)={r['peak_gap_tiers']:.2e}  "
                  f"pic gap(0.35-0.65)={r['peak_gap_035']:.2e}  max hors gap={r['max_out']:.1e}   [{time.time()-t0:.0f}s]",
                  flush=True)
        np.save(f"results/v2_E_{name}_{wlabel.replace('/', '')}.npy", E)
        json.dump(res, open("results/v2_gap.json", "w"), indent=1)
