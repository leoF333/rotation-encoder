"""T3 — Lois d'échelle de E_gen(P) prédites par la théorie (src/aerot/theory.py), grands spectres synthétiques.

Cas A : λ_k = k^-a, t_k = k^-b (loi de puissance).   Cas B : λ_k = e^{-a k}, t_k = e^{-b k} ou k^-b.
Régimes : (i) sans bruit, ridgeless ; (ii) bruit σ², ridgeless ; (iii) bruit σ², ridge optimal.
Prédictions analytiques (derivations/03_generalisation.md §4) :
  A(i)  E ~ P^{-(b-1)}              (si b < 2a+1)
  A(ii) E -> σ²/(1-γ∞), γ∞ = 1 - 1/a  => E -> a σ²     (P << N)   « tempéré »
  A(iii) E-σ² ~ (σ²/P)^{(b-1)/b}
  B(i)  E ~ e^{-b P} (t exp) / P^{-(b-1)} (t puissance)
  B(ii) E ≈ σ² (1 + a P)            « catastrophique »
  B(iii) E-σ² ≈ (σ²/P) ln(P/σ²)/b (t exp)
"""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerot import theory as th

def slope(P, E):
    return np.polyfit(np.log(P), np.log(E), 1)[0]

Nbig = 200000
k = np.arange(1, Nbig + 1).astype(float)
Ps = np.array([100, 200, 400, 800, 1600, 3200])
print("=== Cas A : loi de puissance")
for a, b in [(1.5, 2.0), (2.0, 3.0), (2.0, 4.0), (1.5, 5.0)]:
    lam = k**-a; t = k**-b; c = np.sqrt(t * lam)
    # (i) sans bruit ridgeless
    E = np.array([th.gen_error(lam, c, t.sum(), P, 0.0) for P in Ps])
    pred = -(b - 1) if b < 2 * a + 1 else -2 * a
    print(f"a={a} b={b}  (i)   pente={slope(Ps, E):+.3f}  attendu {pred:+.3f}")
    # (ii) bruit ridgeless
    s2 = 1e-3
    E = np.array([th.gen_error(lam, c, t.sum() + s2, P, 0.0) for P in Ps])
    print(f"             (ii)  E/σ² = {E[-3]/s2:.3f}, {E[-1]/s2:.3f}  attendu a = {a}")
    # (iii) ridge optimal
    Pl = Ps * 16
    Eo = np.array([th.optimal_ridge(lam, c, t.sum() + s2, P)[1] for P in Pl])
    print(f"             (iii) pente(E-σ²)={slope(Pl, Eo - s2):+.3f}  attendu {-(b-1)/b:+.3f}")

print("\n=== Cas B : spectre exponentiel λ_k = e^{-a k}")
kb = np.arange(4000).astype(float)
for a, b, kind in [(0.3, 0.4, "exp"), (0.05, 3.0, "pow")]:
    lam = np.exp(-a * kb); t = np.exp(-b * kb) if kind == "exp" else (1 + kb) ** -b; c = np.sqrt(t * lam)
    Pb = np.array([40, 60, 80, 100]) if kind == "exp" else np.array([320, 640, 1280, 2560])
    E = np.array([th.gen_error(lam, c, t.sum(), P, 0.0) for P in Pb])
    if kind == "exp":
        # prédiction : E ≈ aP Σ_{k>P} t_k ≈ aP e^{-bP}/(1-e^{-b})
        print(f"a={a} b={b} t exp (i) : ln E pente = {np.polyfit(Pb, np.log(E), 1)[0]:+.4f}  attendu -b = {-b:+.4f} ;"
              f" E/(aP·queue) = {np.round(E/(a*Pb*np.array([t[int(P):].sum() for P in Pb])), 2)}")
    else:
        print(f"a={a} b={b} t puiss. (i) : pente log-log = {slope(Pb, E):+.3f}  attendu 2-b = {2-b:+.3f} (amplification aP)")
    s2 = 1e-6
    E = np.array([th.gen_error(lam, c, t.sum() + s2, P, 0.0) for P in Pb])
    E0 = np.array([th.gen_error(lam, c, t.sum(), P, 0.0) for P in Pb])
    print(f"             (ii) (E-E_sans_bruit)/σ² = {np.round((E-E0)/s2, 1)}  attendu ≈ aP = {np.round(a*Pb, 1)}")
    s2 = 1e-3   # régime où le bruit domine le biais sans bruit
    Pb = Pb * 8
    Eo = np.array([th.optimal_ridge(lam, c, t.sum() + s2, P)[1] for P in Pb])
    if kind == "exp":
        print(f"             (iii) (E-σ²)·P/σ² = {np.round((Eo - s2)*Pb/s2, 2)} ; ln(P/σ²)/b = {np.round(np.log(Pb/s2)/b, 2)}")
    else:
        print(f"             (iii) pente(E-σ²) = {slope(Pb, Eo - s2):+.3f}  attendu {-(b-1)/b:+.3f}")
