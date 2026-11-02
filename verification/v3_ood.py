"""V3 : formule des répliques E_OOD / E_gen (ré-implémentée depuis 04 §2.3 et 03) vs simulations ridge sur vraies vidéos.

Entraînement : P vidéos, ω ~ créneau(ω_max = 2π/T) ; W = argmin ½Σ(ω−W·φ)² + λ_r/2 ‖W‖², λ_r = r·P·λ_1.
Théorie : κ(1 − df(κ)/P) = λ_r/P ; W̄ = Σ c_k/(λ_k+κ) u_k ; E_gen = E_tr(W̄)/(1 − γ), γ = Σ L_k²/P ;
E_OOD = E_te(W̄) + (E_gen/P) Σ_k (Σ_te)_kk λ_k/(λ_k+κ)².
"""
import numpy as np, json, sys, time
from scipy.optimize import brentq
import vlib

name = sys.argv[1]
wmax = 2 * np.pi / vlib.T
f = vlib.TEMPLATES[name]
rng = np.random.default_rng(12345)

pop = vlib.population_svd(f, "creneau", wmax, n=4000)
s, U, b = pop["s"], pop["U"], pop["b"]
keep = (s / s[0]) ** 2 > 1e-14
lam, U, c = s[keep] ** 2, U[:, keep], (s * b)[keep]
om_tr, w_tr = pop["nodes"], pop["weights"]
Phi_tr = vlib.render(f, om_tr)
om_te, w_te = vlib.gl_nodes(vlib.support("gap", wmax), 2000)
Phi_te = vlib.render(f, om_te)
Ste_diag = (w_te[:, None] * (Phi_te @ U) ** 2).sum(0)          # (Σ_te)_kk


def pop_err(W, Phi, om, w):
    return float(w @ (om - Phi @ W) ** 2)


def theory(P, lam_r):
    kp = lam_r / P
    df = lambda k: np.sum(lam / (lam + k))
    kappa = brentq(lambda k: k * (1 - df(k) / P) - kp, kp, kp + lam.sum() / P + 1e-300, xtol=1e-16, rtol=1e-12)
    L = lam / (lam + kappa)
    Wbar = U @ (c / (lam + kappa))
    gamma = np.sum(L ** 2) / P
    Egen = pop_err(Wbar, Phi_tr, om_tr, w_tr) / (1 - gamma)
    Eood = pop_err(Wbar, Phi_te, om_te, w_te) + Egen / P * np.sum(Ste_diag * lam / (lam + kappa) ** 2)
    return dict(kappa=kappa, df=df(kappa), Egen=Egen, Eood=Eood)


def sample_creneau(P):
    u = rng.uniform(0, 2 * wmax / 3, P)
    return np.where(u < wmax / 3, u, u + wmax / 3)


rs = [1e-6, 1e-4, 1e-2]
Ps = [(100, 20), (320, 20), (1000, 20), (3000, 8)]
out = []
for P, ntr in Ps:
    t0 = time.time()
    sims = {r: [] for r in rs}
    for _ in range(ntr):
        om = sample_creneau(P)
        X = vlib.render(f, om)
        ev, Q = np.linalg.eigh(X @ X.T)
        Qw = Q.T @ om
        for r in rs:
            lam_r = r * P * lam[0]
            alpha = Q @ (Qw / (ev + lam_r))
            W = X.T @ alpha
            sims[r].append((pop_err(W, Phi_tr, om_tr, w_tr), pop_err(W, Phi_te, om_te, w_te)))
    for r in rs:
        a = np.array(sims[r])
        th = theory(P, r * P * lam[0])
        row = dict(template=name, P=P, r=r, ntrials=ntr, x=th["df"] * np.log(P) / P, df=th["df"],
                   Egen_sim=a[:, 0].mean(), Egen_sim_se=a[:, 0].std() / np.sqrt(ntr), Egen_th=th["Egen"],
                   Eood_sim=a[:, 1].mean(), Eood_sim_se=a[:, 1].std() / np.sqrt(ntr), Eood_th=th["Eood"])
        out.append(row)
        print(f"{name:14s} P={P:5d} r={r:.0e} df={row['df']:7.1f} x={row['x']:6.2f} | "
              f"E_gen sim {row['Egen_sim']:.3e}±{row['Egen_sim_se']:.1e} th {row['Egen_th']:.3e} (×{row['Egen_sim']/row['Egen_th']:.3f}) | "
              f"E_OOD sim {row['Eood_sim']:.3e}±{row['Eood_sim_se']:.1e} th {row['Eood_th']:.3e} (×{row['Eood_sim']/row['Eood_th']:.3f})",
              flush=True)
    json.dump(out, open(f"results/v3_ood_{name}.json", "w"), indent=1)
    print(f"  [P={P} : {time.time()-t0:.0f}s]", flush=True)
