"""T1 — Entropie exacte à alpha>1 (ridge nul) vs formule RS du mémoire (Eq. 21).

Mesure de Gibbs : W ~ N(W_OLS, (beta X^T X)^{-1})  =>  s = S/N exact :
  s = 1/2 + 1/2 ln(2 pi) - 1/2 ln beta - (1/2N) ln det(X^T X)
Avec X = G M^{1/2} et Marchenko-Pastur : (1/N) ln det(G^T G / P) -> -1 + (alpha-1) ln(alpha/(alpha-1)).
=> s = [ (alpha-1)/2 ln(alpha-1) - alpha/2 ln alpha + 1/2 - ln(beta)/2 ]   (= Eq. 21 du mémoire)
       + 1/2 + 1/2 ln(2 pi) - 1/2 ln N - (1/2N) ln det M                   (constantes omises)
"""
import numpy as np
rng = np.random.default_rng(1)
N = 400
for alpha in [1.2, 2.0, 5.0]:
    P = int(alpha*N)
    vals = []
    for t in range(20):
        G = rng.standard_normal((P, N))
        vals.append(np.linalg.slogdet(G.T @ G / P)[1]/N)
    mp = -1 + (alpha-1)*np.log(alpha/(alpha-1))
    print(f"alpha={alpha}: (1/N)E ln det(G^TG/P) sim={np.mean(vals):.4f}  MP={mp:.4f}")
