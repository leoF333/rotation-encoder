"""(a) spectre angulaire de la gaussienne excentrée (03 §6) ; (b) bords francs : quart de disque en forme close ;
(c) lois d'échelle de 03 §5 évaluées sur la formule RS (sur-apprentissage tempéré / catastrophique) ; (d) ridge bayésien."""
import numpy as np
from scipy.special import ive
from scipy.optimize import brentq, minimize_scalar

# (a) A(theta) = int s0(x) s0(R_theta x) dx par quadrature sur grille fine, puis Fourier en theta
sig, r0 = 3.0, 7.8
h = 0.05; g = np.arange(-30, 30, h); X, Y = np.meshgrid(g, g)
s0 = np.exp(-((X - r0)**2 + Y**2)/(2*sig**2))
nth = 256; th = 2*np.pi*np.arange(nth)/nth
A = []
for t in th:   # s0(R_t x) : gaussienne centrée en R_{-t} x0
    cx, cy = r0*np.cos(-t), r0*np.sin(-t)
    A.append(np.sum(s0*np.exp(-((X - cx)**2 + (Y - cy)**2)/(2*sig**2)))*h*h)
A = np.array(A)
Sm_num = np.real(np.fft.fft(A))/nth
z0 = r0**2/(2*sig**2)
Sm_th = np.pi*sig**2*ive(np.arange(nth), z0)
for m in [0, 1, 3, 6, 10, 15, 20]:
    print(f'gauss m={m:2d}: num {Sm_num[m]:.6e}  pi s^2 e^-z0 I_m(z0) = {Sm_th[m]:.6e}  rel {abs(Sm_num[m]/Sm_th[m]-1):.1e}')

# (b) quart de disque (coin au centre) : s_m(r) = (1-e^{-i m pi/2})/(2 pi i m) pour r<R  =>  S_m ∝ (2-2cos(m pi/2))/m^2
m = np.arange(1, 41)
Sq = (2 - 2*np.cos(m*np.pi/2))/m**2
nz = Sq > 1e-12
print('quart de disque : pente log S_m vs log m (m non multiple de 4) =', np.polyfit(np.log(m[nz]), np.log(Sq[nz]), 1)[0],
      '; S_m = 0 pour m ≡ 0 mod 4 :', np.allclose(Sq[m % 4 == 0], 0))

# (c) formule RS : E_gen au point d'interpolation, bruit sigma^2, sans ridge
def Egen(lam, t, s2, P, lr=0.0):
    df = lambda k: np.sum(lam/(lam + k))
    if lr == 0:
        kap = brentq(lambda k: df(k) - P, 1e-300, 1e6) if len(lam) > P else 0.0
    else:
        kap = brentq(lambda k: k*(1 - df(k)/P) - lr/P, 1e-300, 1e6)
    gam = np.sum(lam**2/(lam + kap)**2)/P
    B0 = kap**2*np.sum(t/(lam + kap)**2)
    return (s2 + B0)/(1 - gam)
k = np.arange(1, 200001)
for a in [1.5, 2.0, 3.0]:
    lam = k**-a
    print(f'loi de puissance a={a}: E_gen/sigma^2 (sans prof, ridgeless) P=1000 -> {Egen(lam, 0*lam, 1.0, 1000):.3f} ; attendu a = {a}')
kk = np.arange(1, 400)
for a in [0.3, 0.5]:
    lam = np.exp(-a*kk)
    for P in [40, 80, 120]:   # kappa résolu en log (kappa ~ e^{-aP} sous la tolérance absolue de brentq)
        L = brentq(lambda L: np.sum(lam/(lam + np.exp(L))) - P, -700, 10, xtol=1e-12)
        gam = np.sum(lam**2/(lam + np.exp(L))**2)/P
        print(f'exponentiel a={a} P={P}: E_gen/sigma^2 = {1/(1-gam):.2f} ; aP = {a*P:.1f}')

# (d) professeur isotrope t_k = tau^2 lam_k : ridge optimal = sigma^2/tau^2 quel que soit P
lam = k[:3000]**-1.2; tau2 = 0.7; s2 = 0.05
for P in [50, 300, 1000]:
    r = minimize_scalar(lambda L: Egen(lam, tau2*lam, s2, P, np.exp(L)), bounds=(-8, 4), method='bounded')
    print(f'ridge optimal P={P}: {np.exp(r.x):.4f} ; sigma^2/tau^2 = {s2/tau2:.4f}')
