"""T3 §6 — Spectre de puissance angulaire S_m des templates (continuum), qui fixe le spectre du noyau
K(ω,ω') = Σ_t A(t(ω−ω')), A(θ) = ∫ s0(x) s0(R_θ x) dx = Σ_m S_m e^{imθ}.
Prédictions : blob gaussien (centre à r0, largeur σ) : S_m = π σ² e^{−z0} I_m(z0), z0 = r0²/(2σ²) (exact, continuum) ;
templates à bords francs : A(θ) a un point anguleux en 0 ⇒ S_m ~ m^{-2} ; symétrie d'ordre 4 (croix) ⇒ S_m = 0 si m ≢ 0 mod 4.
"""
import numpy as np
from scipy.special import ive

R, nr, nth = 1.0, 800, 4096
r = (np.arange(nr) + 0.5) * R / nr
th = 2 * np.pi * np.arange(nth) / nth
Rg, Tg = np.meshgrid(r, th, indexing="ij")
X, Y = Rg * np.cos(Tg), Rg * np.sin(Tg)

def S_m(img):
    sm = np.fft.fft(img, axis=1) / nth           # s_m(r)
    return 2 * np.pi * np.sum(np.abs(sm) ** 2 * r[:, None], axis=0) * (R / nr)

# blob gaussien
r0, sig = 0.45, 0.12
S = S_m(np.exp(-((X - r0) ** 2 + Y**2) / (2 * sig**2)))
z0 = r0**2 / (2 * sig**2)
exact = np.pi * sig**2 * ive(np.arange(40), z0)   # ive = e^{-z} I_m(z)
print(f"blob gaussien z0={z0:.2f} : m, S_m num, S_m exact")
for m in [0, 2, 5, 10, 15, 20, 25]:
    print(f"   {m:3d}  {S[m]:.4e}  {exact[m]:.4e}")

def slope(S, ms):
    ms = np.array([m for m in ms if S[m] > 0])
    return np.polyfit(np.log(ms), np.log(S[ms]), 1)[0]

cross = ((np.abs(X) < 0.12) & (np.abs(Y) < 0.8)) | ((np.abs(Y) < 0.12) & (np.abs(X) < 0.8))
S = S_m(cross.astype(float))
ms4 = np.arange(8, 400, 4)
print(f"croix : S_m (m≢0 mod 4) max relatif = {S[1:200][np.arange(1,200) % 4 != 0].max()/S[4]:.1e} ; pente log S_m (m≡0 mod 4, 8..400) = {slope(S, ms4):.2f}")
qd = (X > 0) & (Y > 0) & (Rg < 0.7)
S = S_m(qd.astype(float))
print(f"quart de disque : pente log S_m (m = 5..400) = {slope(S, np.arange(5, 400)):.2f}")
seg = (np.abs(Y) < 0.03) & (X > 0.1) & (X < 0.6)
S = S_m(seg.astype(float))
print(f"segment : pente log S_m (m = 5..400) = {slope(S, np.arange(5, 400)):.2f}")
