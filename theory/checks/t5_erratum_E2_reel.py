"""T5 — Impact de l'erratum E2 (signe de ω̄ dans m*) sur les vrais spectres (loi uniforme).
On reconstruit les J du mémoire (covariance CENTRÉE, correction E1) à partir des moments non centrés par Sherman–Morrison,
avec pseudo-inverse tronquée (λ_k/λ_1 > τ) :  a = φ̄ᵀΣ⁺φ̄, b = φ̄ᵀΣ⁺C, d = CᵀΣ⁺C,
    J_φφ = a/(1−a),  J_φc = (b − ω̄a)/(1−a),  J_cc = (d − 2ω̄b + ω̄²a) + (b − ω̄a)²/(1−a).
Puis X = q̃² + 2m + σ² + c (mémoire corrigé = σ²_res ; mémoire original = X avec le mauvais signe)."""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

for wtag in ["0.6283", "1.2566"]:
    for tpl in ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]:
        z = np.load(os.path.join(ROOT, "data", "spectra", f"{tpl}_uniform_d28_T10_wmax{wtag}.npz"))
        lam, c, pm = z["eigvals"], z["c_proj"], z["phi_mean_proj"]
        wbar, varw, om2 = float(z["omega_mean"]), float(z["omega_var"]), float(z["omega2"])
        out = []
        for tau in [1e-4, 1e-8]:
            k = lam > tau * lam[0]
            a = np.sum(pm[k] ** 2 / lam[k]); b = np.sum(pm[k] * c[k] / lam[k]); d = np.sum(c[k] ** 2 / lam[k])
            Jpp = a / (1 - a); Jpc = (b - wbar * a) / (1 - a)
            Jcc = (d - 2 * wbar * b + wbar**2 * a) + (b - wbar * a) ** 2 / (1 - a)
            mc = th.memoire_ridgeless(Jcc, Jpc, Jpp, wbar, varw, alpha=2.0, corrected=True)
            mo = th.memoire_ridgeless(Jcc, Jpc, Jpp, wbar, varw, alpha=2.0, corrected=False)
            s2 = om2 - d
            out.append((tau, 1 - a, s2, mc["X"], mo["X"]))
        print(f"{tpl:>15} ωmax={wtag}: " + " | ".join(
            f"τ={t:.0e}: 1−a={oa:.1e} σ²_res={s2:.2e} X_corr={xc:.3e} (X_mém−X_corr)/X_corr={(xo-xc)/xc:+.1e}" for t, oa, s2, xc, xo in out))
