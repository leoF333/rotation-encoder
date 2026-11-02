"""Décomposition de Runge recalculée indépendamment sur les spectres exportés (créneau).
Contrôle : identité, rôle des directions mu<2 vs mu>>1, borne corrigée, valeurs E_OOD(tau) du tableau 04 §6."""
import numpy as np, sys, glob, os
D = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'spectra')) + os.sep

def analyse(path, taus=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12)):
    z = np.load(path, allow_pickle=False)
    lam = z['eigvals']; c = z['c_proj']; Kx = int(z['K_export'])
    Ste = z['test_cov_in_train_basis']; cte = z['test_c_proj']; o2te = float(z['test_omega2'])
    name = os.path.basename(path).replace('_d28_T10', '')
    out = []
    for tau in taus:
        K = int(np.sum(lam/lam[0] > tau))
        if K > Kx:
            out.append((tau, K, None)); continue
        l = lam[:K]; cc = c[:K]; S = Ste[:K, :K]; ct = cte[:K]
        W = cc/l
        Eood = o2te - 2*ct @ W + W @ S @ W
        # spectre généralisé (S_te, diag l)
        Dm = 1/np.sqrt(l)
        M = (Dm[:, None]*S)*Dm[None, :]
        mu, Q = np.linalg.eigh((M + M.T)/2)
        V = Dm[:, None]*Q
        Su = 2/3*np.diag(l) + S/3; Cu = 2/3*cc + ct/3
        Wu = np.linalg.solve(Su, Cu)
        e = V.T @ (cc - l*Wu)
        b = V.T @ ct
        pos = mu > 0
        Emin = o2te - np.sum(b[pos]**2/mu[pos])
        terms = (mu + 2)**2/np.where(pos, mu, np.nan)*e**2
        rhs = Emin + np.nansum(terms)
        eu_tr = np.sum(e**2)  # = ||proj eps_u||^2 sur F (Bessel)
        share_big = np.nansum(terms[mu > 2])/np.nansum(terms)
        share_mu1 = np.nansum(terms[mu > 1])/np.nansum(terms)
        out.append((tau, K, dict(E=Eood, rhs=rhs, Emin=Emin, mumax=mu.max(), mumin=mu.min(), nbig=int(np.sum(mu > 1)),
                                 share_big=share_big, share_mu1=share_mu1)))
    print(name, f'K_export={Kx}')
    for tau, K, r in out:
        if r is None:
            print(f'   tau={tau:.0e} K={K:5d}  (> K_export)'); continue
        print(f"   tau={tau:.0e} K={K:5d}  E_OOD={r['E']:.2e}  identité: écart rel {abs(r['rhs']-r['E'])/r['E']:.1e}"
              f"  E_min/E={max(r['Emin'],0)/r['E']:.1e}  mu_max={r['mumax']:.1e}  #mu>1={r['nbig']:4d}"
              f"  part(mu>2)={r['share_big']:.3f}")

for f in sorted(glob.glob(D + '*_creneau_*.npz')):
    analyse(f)
