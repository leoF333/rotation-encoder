"""Numérisation des courbes E(ω) des Fig. 6–7 du mémoire à partir des images JPEG embarquées dans le PDF.

1. Extraction des flux /DCTDecode (JPEG) de `docs/memoire_PRL_2025-2026.pdf` -> results/memoire_digit/img_XX.jpg (non commités).
2. Pour chaque panneau E(ω) : cadre des axes repéré à la main (pixels), abscisse [-0.0628, 1.3195] (marges matplotlib de 5 %
   sur [0, 4π/T]), ordonnée log de 1e-7 (Fig. 6) ou 1e-6 (Fig. 7) à 10 ; courbe = pixels rouge foncé (bord supérieur).
Sortie : results/memoire_digit/memoire_Eomega_fig6_7.json (ω, log10 E, pic, zone grisée).
Usage : python scripts/a5_digitize_memoir.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from aerot.paths import ROOT, RES

OUT = RES / "memoire_digit"
# (image, template, gauche, droite, haut, bas, (ymin, ymax)) — cadres mesurés sur les images extraites
PANELS = [("07", "gaussian", 133, 1353, 47, 698, (1e-7, 10)), ("07", "segment", 1474, 2694, 47, 698, (1e-7, 10)),
          ("09", "quarter_circle", 92, 1290, 51, 721, (1e-6, 10)), ("09", "cross", 1393, 1393 + 1198, 51, 721, (1e-6, 10))]
X0, X1 = -0.0628, 1.3195


def extract():
    b = (ROOT / "docs" / "memoire_PRL_2025-2026.pdf").read_bytes()
    for i, m in enumerate(re.finditer(rb"/Subtype\s*/Image", b)):
        st = b.rfind(b"<<", 0, m.start())
        s = b.find(b"stream", m.start()) + 6
        while b[s:s + 1] in (b"\r", b"\n"):
            s += 1
        L = int(re.search(rb"/Length\s+(\d+)", b[st:s]).group(1))
        (OUT / f"img_{i:02d}.jpg").write_bytes(b[s:s + L])


def digitize():
    res = {}
    for img, name, left, right, top, bot, ylim in PANELS:
        a = np.asarray(Image.open(OUT / f"img_{img}.jpg").convert("RGB")).astype(int)
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        red = (R - G > 60) & (R - B > 60) & (G < 120)
        xs, ys = [], []
        for c in range(left + 4, min(right - 3, a.shape[1])):
            r = np.nonzero(red[top + 3:bot - 2, c])[0]
            if len(r):
                yv = r.min() + top + 3
                xs.append(X0 + (c - left) / (right - left) * (X1 - X0))
                ys.append(np.log10(ylim[1]) - (yv - top) / (bot - top) * (np.log10(ylim[1]) - np.log10(ylim[0])))
        xs, ys = np.array(xs), np.array(ys)
        row = a[(top + bot) // 2 + 40]
        pink = np.nonzero((row[:, 0] > 235) & (row[:, 0] - row[:, 1] > 12) & (row[:, 1] > 180))[0]
        pink = pink[(pink > left) & (pink < right)]
        gl = X0 + (pink.min() - left) / (right - left) * (X1 - X0)
        gr = X0 + (pink.max() - left) / (right - left) * (X1 - X0)
        i = int(np.argmax(ys))
        res[name] = dict(omega=xs.tolist(), log10E=ys.tolist(), ylim=ylim, gap_shade=[gl, gr],
                         peak_log10E=float(ys[i]), peak_omega=float(xs[i]),
                         note="numérisé depuis l'image JPEG embarquée du PDF ; cadre = [-0.0628, 1.3195] "
                              "(marges matplotlib de 5 % sur [0, 4π/T])")
        print(f"{name}: pic {10 ** ys[i]:.2e} en ω = {xs[i]:.3f}, zone grisée [{gl:.3f}, {gr:.3f}]")
    (OUT / "memoire_Eomega_fig6_7.json").write_text(json.dumps(res))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    extract()
    digitize()
