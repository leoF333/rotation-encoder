"""T1 — Vérification symbolique (sympy) de l'algèbre RS du mémoire (annexe p. 25-29).

Notations du mémoire : J_cc, J_phic, J_phiphi (= S_.. p.29), wbar (= omega barre), s2 (= sigma_omega^2).
A = [[Jcc, Jphic],[Jphic, Jphiphi]], B = A^{-1}.
"""
import sympy as sp

Jcc, Jpc, Jpp, wbar, s2 = sp.symbols('J_cc J_phic J_phiphi wbar sigma2', real=True)
m, qt, q0, dq, al, be, n = sp.symbols('m qtilde q0 deltaq alpha beta n', real=True)

A = sp.Matrix([[Jcc, Jpc], [Jpc, Jpp]])
B = A.inv()
wt = qt - wbar
c = B[0, 0]*m**2 + 2*B[0, 1]*m*wt + B[1, 1]*wt**2

# --- 1. Développement n -> 0 de l'action RS (p.26-27) -----------------------
qi0 = q0 - 2*m - s2
S_n = (be*al/2*qt**2*n/(1 + be*dq + n*be*q0)
       + al/2*(n - 1)*sp.log(1 + be*dq) + al/2*sp.log(1 + be*dq + n*be*q0)
       - sp.Rational(1, 2)*(n - 1)*sp.log(dq) - sp.Rational(1, 2)*sp.log(dq + n*(qi0 - c)))
S1 = sp.series(S_n, n, 0, 2).removeO().coeff(n, 1)
S1_memoire = (al*be/2*(qt**2 + q0)/(1 + be*dq) + al/2*sp.log(1 + be*dq)
              - sp.log(dq)/2 - (q0 - 2*m - s2 - c)/(2*dq))
print('[1] Ordre n de S_RS/N == Eq.(14) :', sp.simplify(S1 - S1_memoire) == 0)

# --- 2. Equations du col ---------------------------------------------------
S = S1_memoire
eq_q0 = sp.diff(S, q0)
dq_star = sp.solve(eq_q0, dq)
print('[2] dS/dq0=0 ->  deltaq* =', dq_star)
dq_s = 1/(be*(al - 1))
eq_m = sp.simplify(sp.diff(S, m).subs(dq, dq_s))
eq_qt = sp.simplify(sp.diff(S, qt).subs(dq, dq_s))
sol = sp.solve([eq_m, eq_qt], [m, qt], dict=True)[0]
qt_s = sp.simplify(sol[qt]); m_s = sp.simplify(sol[m])
print('    qtilde* =', qt_s)
print('    m*      =', m_s)
qt_mem = (wbar - Jpc)/(1 + Jpp)
m_mem = Jpc*(Jpc + wbar)/(1 + Jpp) - Jcc
print('    qtilde* == memoire Eq.(18) :', sp.simplify(qt_s - qt_mem) == 0)
print('    m*      == memoire Eq.(18) :', sp.simplify(m_s - m_mem) == 0)
print('    m* - m_memoire =', sp.factor(sp.simplify(m_s - m_mem)))

# c au col (Eq.13 vs p.29)
c_p29 = (Jpp*m**2 - 2*Jpc*m*(qt - wbar) + Jcc*(qt - wbar)**2)/(Jcc*Jpp - Jpc**2)
print('[3] c Eq.(13) == c p.29 :', sp.simplify(c - c_p29) == 0)

# --- 3. q0* ------------------------------------------------------------------
eq_dq = sp.diff(S, dq).subs(dq, dq_s)
q0_s = sp.solve(eq_dq, q0)[0]
q0_mem = (qt**2 + al*(2*m + s2 + c))/(al - 1)
print('[4] q0* == Eq.(19) (m,qtilde libres) :', sp.simplify(q0_s - q0_mem) == 0)

# --- 4. Energie libre et quantité clé X = qtilde^2 + 2m + s2 + c ------------
f = sp.simplify(S.subs(dq, dq_s)/be)
f_mem = (al - 1)/2*(qt**2 + 2*m + s2 + c) + al/(2*be)*sp.log(al/(al - 1)) + sp.log((al - 1)*be)/(2*be)
print('[5] f == Eq.(20) (q0 disparait) :', sp.simplify(sp.expand_log(f - f_mem, force=True)) == 0)

X_true = sp.simplify((qt**2 + 2*m + s2 + c).subs({m: m_s, qt: qt_s}))
X_mem = sp.simplify((qt**2 + 2*m + s2 + c).subs({m: m_mem, qt: qt_mem}))
sig_res = s2 - Jcc + (wbar - Jpc)**2/(1 + Jpp)
print('[6] X(col exact) == sigma_res^2 = s2 - Jcc + (wbar-Jphic)^2/(1+Jphiphi) :',
      sp.simplify(X_true - sig_res) == 0)
print('    X(avec m* du memoire) - sigma_res^2 =', sp.factor(sp.simplify(X_mem - sig_res)))

# --- 5. Entropie ---------------------------------------------------------------
fb = f_mem
s_ent = sp.simplify(be**2*sp.diff(fb, be))
s_mem = (al - 1)/2*sp.log(al - 1) - al/2*sp.log(al) + sp.Rational(1, 2) - sp.log(be)/2
print('[7] entropie == Eq.(21) :', sp.simplify(sp.expand_log(s_ent - s_mem, force=True)) == 0)
u = sp.simplify(sp.diff(be*fb, be))
print('    energie <eps> = d(beta f)/dbeta =', sp.simplify(u))

# --- 6. Generalisation : q0* + qtilde*^2 au col ------------------------------
Egen = sp.simplify((q0_mem + qt**2).subs({m: m_s, qt: qt_s}))
print('[8] q0* + qtilde*^2 == alpha/(alpha-1) sigma_res^2 :', sp.simplify(Egen - al/(al - 1)*sig_res) == 0)
