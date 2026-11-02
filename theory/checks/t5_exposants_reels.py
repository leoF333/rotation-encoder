"""T5 — Exposants spectraux des vrais templates (loi uniforme) et lois d'échelle prédites (03 §5).
Ajuste λ_k ~ k^{-a} et la puissance du professeur cumulée Σ_{j>k} t_j ~ k^{-(b-1)} sur une plage de modes (bords francs),
ou ln λ_k ~ -k (lisses) ; compare à la courbe théorique E_gen(P) ridgeless/petit ridge (pente locale)."""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

for wtag in ["0.6283", "1.2566"]:
    for tpl in ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]:
        z = np.load(os.path.join(ROOT, "data", "spectra", f"{tpl}_uniform_d28_T10_wmax{wtag}.npz"))
        lam = z["eigvals"]; k = lam > 1e-14 * lam[0]; lam = lam[k]; c = z["c_proj"][k]; om2 = float(z["omega2"])
        t = c**2 / lam
        tail = np.cumsum(t[::-1])[::-1]            # Σ_{j≥k} t_j
        K = lam.size
        kk = np.arange(1, K + 1)
        if K > 200:
            m = (kk >= 20) & (kk <= min(600, K // 2))
            a = -np.polyfit(np.log(kk[m]), np.log(lam[m] / lam[0]), 1)[0]
            m2 = m & (tail > 1e-14 * om2)
            bm1 = -np.polyfit(np.log(kk[m2]), np.log(tail[m2]), 1)[0]
            desc = f"loi de puissance a={a:.2f}, queue Σt ~ k^-{bm1:.2f} (b={bm1+1:.2f})"
        else:
            m = (kk >= 3) & (lam > 1e-12 * lam[0])
            s = -np.polyfit(kk[m], np.log(lam[m] / lam[0]), 1)[0]
            desc = f"décroissance ~exp(-{s:.2f} k) (K(1e-12)={int(np.sum(lam > 1e-12*lam[0]))})"
        Ps = np.array([100, 300, 1000, 3000])
        E = [th.gen_error(lam, c, om2, P, 1e-10 * P * lam[0]) for P in Ps]
        sl = np.polyfit(np.log(Ps), np.log(E), 1)[0]
        print(f"{tpl:>15} ωmax={wtag}: {desc} ; E_gen(P, r=1e-10) = " + ", ".join(f"{e:.1e}" for e in E) + f" ; pente {sl:.2f}")
