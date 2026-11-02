"""T2 — Vérification de la solution RS avec ridge (modèle gaussien centré, spectre anisotrope).

Modèle : phi ~ N(0, Sigma), Sigma = diag(lam_k) (base propre), omega = theta.phi + eps, eps ~ N(0, s2).
H(W) = 1/2 sum_mu (omega_mu - W.phi_mu)^2 + lam_r/2 |W|^2 ,  Z = int dW exp(-beta H).

Prédictions RS (voir derivations/02_ridge_entrainement.md) :
  chi = sum_k lam_k / (lam_k P/(1+chi) + lam_r)     (chi = beta*deltaQ, indépendant de beta)
  kappa = lam_r (1+chi)/P  <=>  kappa (1 - df(kappa)/P) = lam_r/P,  df = sum lam/(lam+kappa)
  gamma = (1/P) sum lam^2/(lam+kappa)^2 ,  B0 = kappa^2 sum lam theta^2/(lam+kappa)^2
  E_gen(T=0) = (s2 + B0)/(1-gamma) ;  E_train(T=0) = (1-df/P)^2 E_gen
  finite beta : E_gen += chi/beta ; E_train += chi/(beta(1+chi))
  <ln Z> = extremum de G_RS (+ N/2 ln 2pi)
"""
import numpy as np
from scipy.optimize import brentq

def solve_kappa(lam, P, lam_r):
    h = lambda lk: np.exp(lk) - lam_r/P - np.exp(lk)*np.sum(lam/(lam+np.exp(lk)))/P
    return np.exp(brentq(h, np.log(lam_r/P)-1, np.log(lam_r/P + lam.sum()/P)+1, xtol=1e-14))

def rs(lam, theta, s2, P, lam_r, beta):
    kap = solve_kappa(lam, P, lam_r)
    df = np.sum(lam/(lam+kap)); gam = np.sum(lam**2/(lam+kap)**2)/P
    chi = df/(P-df)
    B0 = kap**2*np.sum(lam*theta**2/(lam+kap)**2)
    Eg0 = (s2+B0)/(1-gam); Et0 = (1-df/P)**2*Eg0
    Q0 = Eg0 - s2; dQ = chi/beta
    R = P*beta/(1+chi); qh = P*beta**2*(Q0+s2)/(1+chi)**2
    a = lam*R + beta*lam_r
    G = (0.5*len(lam)*np.log(2*np.pi) + 0.5*(R*(dQ+Q0) - qh*dQ)
         - 0.5*P*(np.log(1+beta*dQ) + beta*(Q0+s2)/(1+beta*dQ))
         + np.sum(-0.5*np.log(a) + 0.5*(lam*qh + beta**2*lam_r**2*theta**2)/a - 0.5*beta*lam_r*theta**2))
    return dict(kappa=kap, df=df, gamma=gam, chi=chi, Egen=Eg0 + chi/beta,
                Etrain=Et0 + chi/(beta*(1+chi)), lnZ=G, Egen0=Eg0, Etrain0=Et0)

rng = np.random.default_rng(2)
N = 300
k = np.arange(1, N+1)
lam = 5*k**-1.5
theta = k**-0.3 * rng.choice([-1, 1], N) / np.sqrt(lam) * 0.3   # puissance t_k = lam theta^2 ~ k^-0.6
s2 = 0.1
beta = 3.0
print(f"{'P':>5} {'lam_r':>7} | {'Etrain sim':>18} {'RS':>9} | {'Egen sim':>18} {'RS':>9} | {'<lnZ>/N sim':>12} {'RS':>9}")
for P in [90, 300, 600]:
    for lam_r in [0.03, 1.0, 10.0]:
        et, eg, lz = [], [], []
        for t in range(60):
            X = rng.standard_normal((P, N))*np.sqrt(lam)
            y = X @ theta + np.sqrt(s2)*rng.standard_normal(P)
            A = X.T @ X + lam_r*np.eye(N)
            W = np.linalg.solve(A, X.T @ y)
            Hmin = 0.5*np.sum((y - X@W)**2) + 0.5*lam_r*W@W
            Ainv_tr = np.trace(np.linalg.solve(A, np.diag(lam)))   # Tr(Sigma A^{-1}) = beta*deltaQ
            # erreurs thermiques moyennées sur Gibbs : E_gen = s2 + (W-th)Sig(W-th) + Tr(Sig A^-1)/beta
            eg.append(s2 + (W-theta)@(lam*(W-theta)) + Ainv_tr/beta)
            # E_train thermique : (1/P)[|y-XW|^2 + Tr(X A^-1 X^T)/beta]
            et.append((np.sum((y-X@W)**2) + np.trace(X @ np.linalg.solve(A, X.T))/beta)/P)
            lz.append(-beta*Hmin + 0.5*N*np.log(2*np.pi/beta) - 0.5*np.linalg.slogdet(A)[1])
        r = rs(lam, theta, s2, P, lam_r, beta)
        f = lambda v: f"{np.mean(v):9.5f}±{np.std(v)/np.sqrt(len(v)):.5f}"
        print(f"{P:5d} {lam_r:7.2f} | {f(et)} {r['Etrain']:9.5f} | {f(eg)} {r['Egen']:9.5f} | "
              f"{np.mean(lz)/N:12.5f} {r['lnZ']/N:9.5f}")
