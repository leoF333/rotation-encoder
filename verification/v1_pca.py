import numpy as np, json, time
from scipy.optimize import least_squares
import vlib

# Contrôle du rendu : disque centré invariant par rotation => Σ de rang 1
disk = lambda x, y: (x * x + y * y < 64).astype(float)
Phi = vlib.render(disk, np.linspace(0, 1.2, 300))
s = np.linalg.svd(Phi, compute_uv=False)
print(f"disque : s2/s1 = {s[1]/s[0]:.2e}  (attendu ~0)")

def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

def fits(om, z):
    lin = r2(z, np.polyval(np.polyfit(om, z, 1), om))
    best = (-np.inf, None)
    for b in np.linspace(0.2, 40, 4000):          # grille fine sur la fréquence, puis raffinement
        X = np.stack([np.sin(b * om), np.cos(b * om), np.ones_like(om)], 1)
        coef = np.linalg.lstsq(X, z, rcond=None)[0]
        v = r2(z, X @ coef)
        if v > best[0]:
            best = (v, (b, coef))
    b, coef = best[1]
    res = least_squares(lambda p: p[1] * np.sin(p[0] * om) + p[2] * np.cos(p[0] * om) + p[3] - z,
                        [b, *coef])
    p = res.x
    return lin, r2(z, p[1] * np.sin(p[0] * om) + p[2] * np.cos(p[0] * om) + p[3]), p[0]

rng = np.random.default_rng(0)
out = {}
for name in ["arc", "gaussian", "quarter_circle"]:
    om = np.sort(rng.uniform(0.05, 2 * np.pi / vlib.T, 2000))
    X = vlib.render(vlib.TEMPLATES[name], om)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    z = Xc @ Vt[0]
    lin, sin_, b = fits(om, z)
    out[name] = dict(R2_lin=lin, R2_sin=sin_, freq=b)
    print(f"{name:15s} R² lin = {lin:.4f}   R² sin = {sin_:.5f}   (b = {b:.2f})")
json.dump(out, open("results/v1_pca.json", "w"), indent=1)
