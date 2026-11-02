# B2 — forme de la queue spectrale (λ_k/λ_1 ∈ [1e-8, 1e-2]) : exponentielle vs loi de puissance

Modèles ajustés sur log10(λ_k/λ_1) : exp = a − b k ; puissance = a − p log10 k ; étirée = a − b k^β. ΔAIC = AIC(exp) − AIC(puissance) (> 0 : la loi de puissance est préférée). « pixel » : spectres exportés (rendu analytic ss=4, d = 28) ; « continuum » : modèle sans pixels (d → ∞, features exactes de continuum.py), restreint aux k < n/2 (saturation du rang de la quadrature n = 2000 au-delà).

| template | ρ | ω_max | rendu | K(1e-4) | K(1e-8) | modèle retenu | b (exp) | p (puissance) | β (étirée) | ΔAIC exp−puis. |
|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | uniform | 2π/T | pixel | 14 | 25 | stretched | 0.353 | 12.8 | 1.34 | -46 |
| gaussian | uniform | 2π/T | continuum | 14 | 25 | stretched | 0.351 | 12.7 | 1.34 | -47 |
| gaussian | uniform | 4π/T | pixel | 26 | 45 | stretched | 0.202 | 12.4 | 1.25 | -63 |
| gaussian | uniform | 4π/T | continuum | 26 | 45 | stretched | 0.2 | 12.3 | 1.24 | -63 |
| gaussian | creneau | 2π/T | pixel | 12 | 21 | stretched | 0.417 | 12.3 | 1.34 | -21 |
| gaussian | creneau | 4π/T | pixel | 19 | 35 | stretched | 0.259 | 12.5 | 1.23 | -60 |
| dipole | uniform | 2π/T | pixel | 13 | 25 | stretched | 0.359 | 13 | 1.32 | -21 |
| dipole | uniform | 2π/T | continuum | 14 | 25 | stretched | 0.356 | 12.9 | 1.31 | -21 |
| dipole | uniform | 4π/T | pixel | 23 | 43 | exp | 0.214 | 13.5 | 1.16 | -22 |
| dipole | uniform | 4π/T | continuum | 23 | 43 | exp | 0.212 | 13.4 | 1.15 | -22 |
| dipole | creneau | 2π/T | pixel | 12 | 20 | stretched | 0.419 | 11.9 | 1.34 | -24 |
| dipole | creneau | 4π/T | pixel | 18 | 33 | exp | 0.27 | 12.6 | 1.15 | -36 |
| quarter_circle | uniform | 2π/T | pixel | 40 | 1265 | power | 0.00231 | 2.37 | 0.02 | +2979 |
| quarter_circle | uniform | 2π/T | continuum | 81 | 1723 | power | 0.00237 | 1.96 | 0.02 | +5763 |
| quarter_circle | uniform | 4π/T | pixel | 66 | 1722 | power | 0.00177 | 2.49 | 0.02 | +3888 |
| quarter_circle | uniform | 4π/T | continuum | 130 | 2000 | power | 0.00226 | 1.92 | 0.02 | +4560 |
| quarter_circle | creneau | 2π/T | pixel | 30 | 1004 | power | 0.00286 | 2.34 | 0.02 | +2278 |
| quarter_circle | creneau | 4π/T | pixel | 50 | 1374 | power | 0.0022 | 2.46 | 0.02 | +3124 |
| cross | uniform | 2π/T | pixel | 51 | 1391 | power | 0.00214 | 2.43 | 0.02 | +2633 |
| cross | uniform | 2π/T | continuum | 140 | 1560 | power | 0.00227 | 1.93 | 0.02 | +5405 |
| cross | uniform | 4π/T | pixel | 69 | 1455 | power | 0.00209 | 2.57 | 0.02 | +2645 |
| cross | uniform | 4π/T | continuum | 176 | 1963 | power | 0.00214 | 1.89 | 0.02 | +3821 |
| cross | creneau | 2π/T | pixel | 41 | 1298 | power | 0.00225 | 2.36 | 0.02 | +2561 |
| cross | creneau | 4π/T | pixel | 60 | 1416 | power | 0.00215 | 2.52 | 0.02 | +2739 |
| segment | uniform | 2π/T | pixel | 75 | 1654 | stretched | 0.0018 | 2.41 | 0.13 | +1609 |
| segment | uniform | 2π/T | continuum | 371 | 1751 | power | 0.0021 | 1.93 | 0.02 | +4375 |
| segment | uniform | 4π/T | pixel | 128 | 2434 | stretched | 0.00132 | 2.61 | 0.18 | +1717 |
| segment | uniform | 4π/T | continuum | 579 | 2000 | power | 0.00192 | 1.9 | 0.02 | +3334 |
| segment | creneau | 2π/T | pixel | 58 | 1297 | stretched | 0.00233 | 2.42 | 0.22 | +867 |
| segment | creneau | 4π/T | pixel | 94 | 1750 | stretched | 0.0018 | 2.56 | 0.19 | +1184 |
| arc | uniform | 2π/T | pixel | 48 | 589 | power | 0.00568 | 2.86 | 0.02 | +1201 |
| arc | uniform | 2π/T | continuum | 81 | 1728 | power | 0.00237 | 1.95 | 0.02 | +5710 |
| arc | uniform | 4π/T | pixel | 83 | 776 | power | 0.0046 | 3.08 | 0.02 | +1729 |
| arc | uniform | 4π/T | continuum | 130 | 2000 | power | 0.00226 | 1.92 | 0.02 | +4504 |
| arc | creneau | 2π/T | pixel | 37 | 430 | power | 0.00735 | 2.78 | 0.02 | +785 |
| arc | creneau | 4π/T | pixel | 61 | 626 | power | 0.00563 | 3.03 | 0.02 | +1326 |
