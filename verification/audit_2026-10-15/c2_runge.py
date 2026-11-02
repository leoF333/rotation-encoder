"""Identité de Runge (04 §4.1) et borne de « Lebesgue » : vérification indépendante sur une vraie variété 1D.
Features : phi_k(w) = cos(pi k w) * s_k (k<K), w in [0,1] ; train = créneau, test = gap, cible y = w (ou autre)."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

rng = np.random.default_rng(0)

def quad(a, b, n=400):
    x, w = leggauss(n)
    return 0.5*(b-a)*x + 0.5*(b+a), 0.5*(b-a)*w

def moments(feat, y, segs):
    Ws, Ys, Fs = [], [], []
    tot = sum(b-a for a, b in segs)
    for a, b in segs:
        x, w = quad(a, b)
        Fs.append(feat(x)); Ys.append(y(x)); Ws.append(w/tot)
    F = np.vstack(Fs); Y = np.concatenate(Ys); Wt = np.concatenate(Ws)
    S = (F*Wt[:, None]).T @ F
    C = (F*Wt[:, None]).T @ Y
    o2 = (Wt*Y**2).sum()
    return S, C, o2

def run(feat, y, label):
    tr = [(0, 1/3), (2/3, 1)]; te = [(1/3, 2/3)]; un = [(0, 1)]
    Str, Ctr, o2tr = moments(feat, y, tr)
    Ste, Cte, o2te = moments(feat, y, te)
    Su, Cu, o2u = moments(feat, y, un)
    # f_tr, f_u dans F (plein rang ici)
    Wtr = np.linalg.solve(Str, Ctr)
    Wu = np.linalg.solve(Su, Cu)
    Ete = lambda W: o2te - 2*Cte @ W + W @ Ste @ W
    Etr_err = lambda W: o2tr - 2*Ctr @ W + W @ Str @ W
    E_ood = Ete(Wtr)
    Emin = o2te - Cte @ np.linalg.solve(Ste, Cte)
    mu, V = eigh(Ste, Str)              # V^T Str V = I, V^T Ste V = diag(mu)
    # e_j = <eps_u, g_j>_tr = <y - Wu.phi, v_j.phi>_tr = V^T (Ctr - Str Wu)
    e = V.T @ (Ctr - Str @ Wu)
    rhs = Emin + np.sum((mu + 2)**2/mu * e**2)
    eu2_tr = Etr_err(Wu); eu2_te = Ete(Wu)
    bound_memo = Emin + (mu.max() + 2)**2/mu.max()*eu2_tr
    bound_corr = Emin + (mu.max() + 2)**2/mu.max()*eu2_tr + 4*eu2_te
    print(f'{label}: E_OOD={E_ood:.6e}  identité={rhs:.6e}  écart rel={abs(rhs-E_ood)/E_ood:.1e}')
    print(f'    mu in [{mu.min():.2e},{mu.max():.2e}]  borne 04§4.1={bound_memo:.3e} ({"OK" if E_ood <= bound_memo*(1+1e-9) else "VIOLÉE"})'
          f'  borne corrigée={bound_corr:.3e} ({"OK" if E_ood <= bound_corr*(1+1e-9) else "VIOLÉE"})')
    small = mu < 2
    print(f'    part des directions mu<2 dans la somme : {np.sum(((mu+2)**2/mu*e**2)[small])/np.sum((mu+2)**2/mu*e**2):.2f}')

for K in [4, 8, 14]:
    s = np.exp(-0.3*np.arange(K))
    run(lambda x, s=s, K=K: np.cos(np.pi*np.outer(x, np.arange(K)))*s, lambda x: x, f'cos K={K}, y=w')

# cas construit pour mettre en défaut la borne : peu de modes, résidu concentré dans le gap
for K in [2, 3]:
    run(lambda x, K=K: np.vstack([(x < 1/3).astype(float)*0 + 1.0] + [np.cos(np.pi*k*x) for k in range(1, K)]).T,
        lambda x: np.exp(-((x-0.5)/0.08)**2), f'contre-exemple K={K}, y=bosse dans le gap')
