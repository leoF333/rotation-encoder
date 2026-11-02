import sympy as sp
Jcc, Jpc, Jpp, wb, s2, a, b = sp.symbols("J_cc J_phic J_phiphi omegabar sigma2 alpha beta", real=True)
m, qt, q0, dq = sp.symbols("m qtilde q0 deltaq", real=True)
A = sp.Matrix([[Jcc, Jpc], [Jpc, Jpp]]); Ai = A.inv()
wt = qt - wb
c = Ai[0, 0] * m**2 + 2 * Ai[0, 1] * m * wt + Ai[1, 1] * wt**2
# Action RS par réplique et par N (Eq. 14 du mémoire)
S = a * b / 2 * (qt**2 + q0) / (1 + b * dq) + a / 2 * sp.log(1 + b * dq) - sp.log(dq) / 2 \
    - (q0 - 2 * m - s2 - c) / (2 * dq)
# équations du col, résolues indépendamment
sol_dq = sp.solve(sp.diff(S, q0), dq)
print("dq* =", sol_dq)
Sd = S.subs(dq, sol_dq[0])
sol = sp.solve([sp.diff(S, m).subs(dq, sol_dq[0]), sp.diff(S, qt).subs(dq, sol_dq[0])], [m, qt], dict=True)[0]
qt_s, m_s = sp.simplify(sol[qt]), sp.simplify(sol[m])
print("qtilde* =", sp.factor(qt_s))
print("m*      =", sp.factor(m_s))
m_memoire = Jpc * (Jpc + wb) / (1 + Jpp) - Jcc
m_corr = Jpc * (Jpc - wb) / (1 + Jpp) - Jcc
print("m* == version mémoire (+omegabar) ?", sp.simplify(m_s - m_memoire) == 0)
print("m* == version corrigée   (-omegabar) ?", sp.simplify(m_s - m_corr) == 0)
X = sp.simplify((qt**2 + 2 * m + s2 + c).subs({qt: qt_s, m: m_s}))
sres = s2 - Jcc + (wb - Jpc)**2 / (1 + Jpp)
print("qt²+2m+σ²+c == σ²_res ?", sp.simplify(X - sres) == 0)
# q0* par ∂S/∂δq = 0, puis E_gen = q0 + qt² (β→∞)
eq = sp.diff(S, dq).subs({qt: qt_s, m: m_s}).subs(dq, sol_dq[0])
q0s = sp.solve(eq, q0)[0]
print("E_gen = q0*+qt*² == α/(α-1) σ²_res ?", sp.simplify(q0s + qt_s**2 - a / (a - 1) * sres) == 0)
# énergie par dimension e = ∂β(βf), f = S*/β ; E_train = 2e/α à β→∞
Sstar = S.subs({qt: qt_s, m: m_s}).subs(q0, q0s).subs(dq, sol_dq[0])
e = sp.diff(sp.simplify(Sstar), b)
e_inf = sp.limit(sp.simplify(e), b, sp.oo)
print("E_train(β→∞) = 2e/α :", sp.simplify(2 * e_inf / a - (1 - 1 / a) * sres) == 0)
