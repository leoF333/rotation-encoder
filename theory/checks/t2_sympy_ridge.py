"""T2 — Vérifications symboliques du calcul RS avec ridge.
[1] développements en n des déterminants / formes quadratiques RS ;
[2] la solution annoncée (chi, R, qh, Q0) annule le gradient de G_RS (K=2 modes symboliques, beta fini) ;
[3] E_train = -(2/P) dG/dbeta_loss  ==  deltaQ/(1+beta deltaQ) + (Q0+s2)/(1+beta deltaQ)^2.
"""
import sympy as sp
n = sp.symbols('n')
a, b = sp.symbols('a b', positive=True)
# [1] ln det(a I + b J) = (n-1) ln a + ln(a+nb) ; 1^T (aI+bJ)^{-1} 1 = n/(a+nb)
ld = (n-1)*sp.log(a) + sp.log(a+n*b)
print('[1] ln det ~ n(ln a + b/a) :', sp.simplify(sp.series(ld, n, 0, 2).removeO() - n*(sp.log(a)+b/a)) == 0)
q = n/(a+n*b)
print('    1^T M^-1 1 ~ n/a        :', sp.simplify(sp.series(q, n, 0, 2).removeO() - n/a) == 0)
# vérif directe sur n=3 entier
M3 = a*sp.eye(3) + b*sp.ones(3, 3)
print('    det, n=3                :', sp.simplify(M3.det() - a**2*(a+3*b)) == 0,
      ' 1^T M^-1 1 :', sp.simplify((sp.ones(1, 3)*M3.inv()*sp.ones(3, 1))[0] - 3/(a+3*b)) == 0)

# [2] G_RS pour K=2 modes
P, beta, lr, s2 = sp.symbols('P beta lambda_r sigma2', positive=True)
Q0, dQ, R, qh = sp.symbols('Q0 deltaQ R qhat', real=True)
l1, l2, t1, t2 = sp.symbols('l1 l2 t1 t2', positive=True)
lam = [l1, l2]; th = [t1, t2]
G = sp.Rational(1, 2)*(R*(dQ+Q0) - qh*dQ) - P/2*(sp.log(1+beta*dQ) + beta*(Q0+s2)/(1+beta*dQ))
for l, t in zip(lam, th):
    A_ = l*R + beta*lr
    G += -sp.log(A_)/2 + (l*qh + beta**2*lr**2*t**2)/(2*A_) - beta*lr*t**2/2
grads = [sp.diff(G, v) for v in (Q0, dQ, R, qh)]
# solution annoncée : chi solution de chi = sum l/(l P/(1+chi) + lr) ; on paramètre par chi et on vérifie
chi = sp.symbols('chi', positive=True)
R_s = P*beta/(1+chi); dQ_s = chi/beta
kap = lr*(1+chi)/P
gam = sum(l**2/(l+kap)**2 for l in lam)/P
B0 = kap**2*sum(l*t**2/(l+kap)**2 for l, t in zip(lam, th))
Q0_s = (gam*s2 + B0)/(1-gam)
qh_s = P*beta**2*(Q0_s+s2)/(1+chi)**2
chi_eq = chi - sum(l/(l*P/(1+chi) + lr) for l in lam)
sub = {R: R_s, dQ: dQ_s, Q0: Q0_s, qh: qh_s}
res = [sp.simplify(g.subs(sub)) for g in grads]
# dG/dqh = -dQ/2 + 1/2 sum l/(lR+beta lr) : proportionnel à chi_eq
print('[2] dG/dQ0 =', res[0], ' ; dG/ddQ =', sp.simplify(res[1]))
print('    dG/dqh / chi_eq =', sp.simplify(res[3]/chi_eq))
# dG/dR : vérifier nullité en éliminant P via l'équation chi (P = ... ) numériquement sur valeurs
import random
ok = True
for trial in range(5):
    vals = {l1: random.uniform(0.1, 3), l2: random.uniform(0.1, 3), t1: random.uniform(-1, 1), t2: random.uniform(-1, 1),
            lr: random.uniform(0.1, 2), s2: random.uniform(0.1, 1), beta: random.uniform(0.5, 5)}
    # choisir P puis résoudre l'équation en chi (racine positive unique)
    from scipy.optimize import brentq
    vals[P] = random.uniform(2.5, 8)
    fchi = sp.lambdify(chi, chi_eq.subs(vals))
    chi_v = brentq(fchi, 1e-12, 1e6)
    v = dict(vals); v[chi] = chi_v
    r2 = sp.N(res[2].subs(v)); r3 = sp.N(res[3].subs(v))
    ok &= abs(r2) < 1e-10 and abs(r3) < 1e-10
print('    dG/dR = dG/dqh = 0 au col (5 tirages numériques) :', ok)

# [3] énergie d'entraînement : beta du terme de perte seulement
bl = sp.symbols('beta_l', positive=True)
Gl = - P/2*(sp.log(1+bl*dQ) + bl*(Q0+s2)/(1+bl*dQ))
Et = sp.simplify(-2/P*sp.diff(Gl, bl))
print('[3] E_train(beta) =', sp.simplify(Et.subs(bl, beta)))
