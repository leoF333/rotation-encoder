"""Contrôle indépendant des formules ridge (02/03) et de la formule OOD des répliques (04 §2.3)
sur données gaussiennes, avec une loi de test de covariance NON diagonale et une cible de test mal spécifiée."""
import numpy as np
from scipy.optimize import brentq
rng = np.random.default_rng(1)

N = 200
lam = np.arange(1, N+1)**-1.3
theta = rng.choice([-1, 1], N)*np.sqrt(0.4*np.arange(1, N+1)**-1.2/lam)   # t_k = lam theta^2 = 0.4 k^-1.2
sig2 = 0.03
# loi de test : covariance non diagonale, cible différente
Bm = np.diag(np.sqrt(lam)); mix = rng.normal(size=(N, N))*0.05
Bm[20:60, 20:60] += mix[20:60, 20:60]*3
Ste = Bm @ Bm.T
theta_te = theta + 0.3*rng.normal(size=N)*np.sqrt(1/np.arange(1, N+1))
Cte = Ste @ theta_te; o2te = theta_te @ Ste @ theta_te + 0.01

def theory(P, lr):
    df = lambda k: np.sum(lam/(lam + k))
    g = lambda k: k*(1 - df(k)/P) - lr/P
    lo = 1e-14 if lr > 0 else brentq(lambda k: df(k) - P, 1e-14, 1e3) if P < N else 0
    kap = brentq(g, max(lo, 1e-14), 1e3) if lr > 0 else lo
    gam = np.sum(lam**2/(lam + kap)**2)/P
    t = lam*theta**2
    B0 = kap**2*np.sum(t/(lam + kap)**2)
    Eg = (sig2 + B0)/(1 - gam)
    Etr = (1 - df(kap)/P)**2*Eg
    Wbar = lam*theta/(lam + kap)
    Ete = lambda W: o2te - 2*Cte @ W + W @ Ste @ W
    Eood = Ete(Wbar) + Eg/P*np.sum(np.diag(Ste)*lam/(lam + kap)**2)
    return Etr, Eg, Eood, kap

def simulate(P, lr, n=300):
    R = []
    for _ in range(n):
        X = rng.normal(size=(P, N))*np.sqrt(lam)
        y = X @ theta + rng.normal(size=P)*np.sqrt(sig2)
        W = np.linalg.solve(X.T @ X + lr*np.eye(N), X.T @ y)
        Etr = np.mean((y - X @ W)**2)
        d = W - theta
        Eg = sig2 + d @ (lam*d)
        Eo = o2te - 2*Cte @ W + W @ Ste @ W
        R.append((Etr, Eg, Eo))
    R = np.array(R)
    return R.mean(0), R.std(0)/np.sqrt(n)

print(f'{"P":>5} {"lam_r":>7} | {"E_train sim":>18} {"th":>9} | {"E_gen sim":>18} {"th":>9} | {"E_OOD sim":>18} {"th":>9}')
for P, lr in [(60, 0.01), (60, 1.0), (150, 0.1), (400, 0.3), (400, 10.0), (1000, 1e-6), (1000, 3.0)]:
    th = theory(P, lr); m, s = simulate(P, lr)
    print(f'{P:5d} {lr:7.0e} | {m[0]:.4f}±{s[0]:.4f} {th[0]:9.4f} | {m[1]:.4f}±{s[1]:.4f} {th[1]:9.4f} | {m[2]:.4f}±{s[2]:.4f} {th[2]:9.4f}')

# déterminant (entropie) : <ln det(X^T X + lr I)> vs équivalent déterministe de 02 §4.3
for P, lr in [(100, 0.5), (400, 2.0)]:
    vals = []
    for _ in range(30):
        X = rng.normal(size=(P, N))*np.sqrt(lam)
        vals.append(np.linalg.slogdet(X.T @ X + lr*np.eye(N))[1])
    _, _, _, kap = theory(P, lr)
    chi = np.sum(lam/(lam + kap))/(P - np.sum(lam/(lam + kap)))
    pred = np.sum(np.log(lr + P*lam/(1 + chi))) + P*np.log(1 + chi) - P*chi/(1 + chi)
    print(f'logdet P={P} lr={lr}: sim {np.mean(vals):.3f} ± {np.std(vals)/np.sqrt(30):.3f}  théorie {pred:.3f}')
