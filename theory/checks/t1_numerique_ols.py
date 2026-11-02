"""T1 — Vérification numérique (données gaussiennes synthétiques) du résultat ridgeless alpha>1.

Modèle : (phi, omega) gaussien joint, moyenne (phibar, wbar), covariance CENTRÉE [[Sc, Cc],[Cc^T, s2]].
Moindres carrés sans biais, sans ridge : W = argmin sum_mu (omega_mu - W.phi_mu)^2.
Prédictions :
  E_train (par échantillon, sans 1/2)  = sigma_res^2 (1 - N/P)
  E_gen   (nouvel échantillon)         = sigma_res^2 / (1 - N/P)       [= P/(P-N) sigma_res^2]
  sigma_res^2 = <omega^2> - C^T Sigma^{-1} C  (moments NON centrés)
              = s2 - Jcc + (wbar - Jphic)^2/(1 + Jphiphi)  (J's avec la covariance CENTRÉE)
Aussi : valeur obtenue avec la formule du mémoire (m* avec le signe erroné) et avec
la matrice non centrée injectée comme covariance (double comptage de la moyenne).
"""
import numpy as np

rng = np.random.default_rng(0)
N = 150
k = np.arange(1, N + 1)
lam_c = k**-1.2                                  # spectre centré anisotrope
U = np.linalg.qr(rng.standard_normal((N, N)))[0]
Sc = (U * lam_c) @ U.T
phibar = 0.3 * U @ (rng.standard_normal(N) * np.sqrt(lam_c))*3 + 0.2*U[:, 0]
theta = U @ (rng.standard_normal(N) * k**-0.5)   # "professeur" dans la base propre
s2_eps = 0.05
wbar = 0.7
# omega = theta.(phi - phibar) + wbar + eps  => cov centrée
Cc = Sc @ theta
s2 = theta @ Sc @ theta + s2_eps

# Moments non centrés
Sig = Sc + np.outer(phibar, phibar)
Cu = Cc + wbar * phibar
om2 = s2 + wbar**2
sig_res = om2 - Cu @ np.linalg.solve(Sig, Cu)

# Formules du mémoire (J avec cov centrée = interprétation cohérente)
Si = np.linalg.inv(Sc)
Jcc, Jpc, Jpp = Cc @ Si @ Cc, phibar @ Si @ Cc, phibar @ Si @ phibar
detA = Jcc*Jpp - Jpc**2
def X_of(m, qt):
    wt = qt - wbar
    c = (Jpp*m**2 - 2*Jpc*m*wt + Jcc*wt**2)/detA
    return qt**2 + 2*m + s2 + c
qt_s = (wbar - Jpc)/(1 + Jpp)
m_ok = Jpc*(Jpc - wbar)/(1 + Jpp) - Jcc
m_mem = Jpc*(Jpc + wbar)/(1 + Jpp) - Jcc
print(f"sigma_res^2 (non centré)          = {sig_res:.6f}")
print(f"X (m* corrigé, cov centrée)       = {X_of(m_ok, qt_s):.6f}")
print(f"X (m* du mémoire, cov centrée)    = {X_of(m_mem, qt_s):.6f}")
# double comptage : on injecte les moments NON centrés comme 'covariance' + moyenne
Si_u = np.linalg.inv(Sig)
Jcc_u, Jpc_u, Jpp_u = Cu @ Si_u @ Cu, phibar @ Si_u @ Cu, phibar @ Si_u @ phibar
X_dc = s2 - Jcc_u + (wbar - Jpc_u)**2/(1 + Jpp_u)
print(f"X (moments non centrés injectés)  = {X_dc:.6f}  (sigma_omega^2 = variance)")
X_dc2 = om2 - Jcc_u + (wbar - Jpc_u)**2/(1 + Jpp_u)
print(f"X (idem avec <omega^2>)           = {X_dc2:.6f}")

Lc = np.linalg.cholesky(Sc)
print("\n alpha   E_train(sim)   pred        E_gen(sim)    pred")
for alpha in [1.25, 1.5, 2.0, 4.0]:
    P = int(alpha*N)
    et, eg = [], []
    for t in range(300):
        Xi = rng.standard_normal((P, N)) @ Lc.T
        X = Xi + phibar
        y = Xi @ theta + wbar + np.sqrt(s2_eps)*rng.standard_normal(P)
        W = np.linalg.lstsq(X, y, rcond=None)[0]
        et.append(np.mean((y - X @ W)**2))
        eg.append(om2 - 2*Cu @ W + W @ Sig @ W)
    et, eg = np.array(et), np.array(eg)
    print(f" {alpha:5.2f}  {et.mean():.5f}±{et.std()/np.sqrt(len(et)):.5f}  {sig_res*(1-N/P):.5f}"
          f"   {eg.mean():.5f}±{eg.std()/np.sqrt(len(eg)):.5f}  {sig_res/(1-N/P):.5f}")
