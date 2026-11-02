"""Vérification indépendante (15/10/2026) du col RS du mémoire, à partir de l'Eq. (14).
On repart de l'action réduite écrite dans le mémoire et on résout les équations du col."""
import sympy as sp

al, be, dq, q0, m, qt, s2, wb = sp.symbols('alpha beta deltaq q0 m qtilde sigma2 omegabar', real=True)
Jcc, Jpc, Jpp = sp.symbols('Jcc Jphic Jphiphi', real=True)
A = sp.Matrix([[Jcc, Jpc], [Jpc, Jpp]])
B = A.inv()
wt = qt - wb
c = B[0, 0]*m**2 + 2*B[0, 1]*m*wt + B[1, 1]*wt**2

S = al*be/2*(qt**2 + q0)/(1 + be*dq) + al/2*sp.log(1 + be*dq) - sp.log(dq)/2 - (q0 - 2*m - s2 - c)/(2*dq)

# col en q0 -> dq
dq_star = sp.solve(sp.diff(S, q0), dq)
print('delta q* =', dq_star)
dqs = dq_star[0]
Sm = S.subs(dq, dqs)
sol = sp.solve([sp.diff(Sm, m), sp.diff(Sm, qt)], [m, qt], dict=True)[0]
m_s = sp.simplify(sol[m]); qt_s = sp.simplify(sol[qt])
print('qtilde* =', sp.factor(qt_s))
print('m*      =', sp.simplify(m_s))
m_memoire = Jpc*(Jpc + wb)/(1 + Jpp) - Jcc
m_corrige = Jpc*(Jpc - wb)/(1 + Jpp) - Jcc
print('m* - m_memoire =', sp.simplify(m_s - m_memoire))
print('m* - m_corrige =', sp.simplify(m_s - m_corrige))

X = (qt**2 + 2*m + s2 + c).subs({m: m_s, qt: qt_s})
sig_res = s2 - Jcc + (wb - Jpc)**2/(1 + Jpp)
print('X - sigma2_res =', sp.simplify(X - sig_res))

# col en dq -> q0
q0_s = sp.solve(sp.diff(S, dq).subs(dq, dqs).subs({m: m_s, qt: qt_s}), q0)[0]
Egen = sp.simplify(q0_s + qt_s**2)
print('q0* + qt*^2 - alpha/(alpha-1) sigma2_res =', sp.simplify(Egen - al/(al - 1)*sig_res))

# énergie libre, énergie, entropie
f = sp.simplify(S.subs(dq, dqs).subs({m: m_s, qt: qt_s, q0: q0_s})/be)
e = sp.simplify(sp.diff(be*f, be))
print('e - [(alpha-1)/2 X + 1/(2 beta)] =', sp.simplify(e - ((al - 1)/2*sig_res + 1/(2*be))))
s = sp.simplify(be**2*sp.diff(f, be))
s21 = (al - 1)/2*sp.log(al - 1) - al/2*sp.log(al) + sp.Rational(1, 2) - sp.log(be)/2
print('s - Eq.21 =', sp.simplify(sp.expand_log(s - s21, force=True)))

# erreur avec le m* du mémoire (E2)
Xm = (qt**2 + 2*m + s2 + c).subs({m: m_memoire, qt: qt_s})
print('X(m_memoire) - sigma2_res =', sp.factor(sp.simplify(Xm - sig_res)))
