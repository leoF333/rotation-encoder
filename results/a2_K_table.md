# Nombre de modes par template (population)

| template | ρ | ω_max | λ_1 | K(1e-4) | K(1e-8) | K(1e-12) | K_c(1e-4) | K_c(1e-12) | modes 99.9 % (non centré) | modes 99.9 % (centré) |
|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | uniform | 2π/T | 144 | 14 | 25 | 34 | 14 | 34 | 11 | 11 |
| dipole | uniform | 2π/T | 333 | 13 | 25 | 33 | 15 | 33 | 11 | 11 |
| quarter_circle | uniform | 2π/T | 409 | 40 | 1265 | 1457 | 46 | 1456 | 36 | 46 |
| cross | uniform | 2π/T | 1.1e+03 | 51 | 1391 | 1410 | 91 | 1409 | 58 | 96 |
| disk | uniform | 2π/T | 1.95e+03 | 1 | 1 | 1 | 996 | 2057 | 1 | 1144 |
| segment | uniform | 2π/T | 22.2 | 75 | 1654 | 1664 | 120 | 1663 | 192 | 264 |
| arc | uniform | 2π/T | 61.5 | 48 | 589 | 590 | 58 | 589 | 46 | 56 |
| gaussian | creneau | 2π/T | 144 | 12 | 21 | 28 | 11 | 29 | 9 | 9 |
| dipole | creneau | 2π/T | 360 | 12 | 20 | 28 | 11 | 28 | 9 | 9 |
| quarter_circle | creneau | 2π/T | 400 | 30 | 1004 | 1110 | 33 | 1109 | 25 | 32 |
| cross | creneau | 2π/T | 1.14e+03 | 41 | 1298 | 1361 | 66 | 1360 | 42 | 71 |
| disk | creneau | 2π/T | 1.95e+03 | 1 | 1 | 1 | 968 | 2005 | 1 | 1254 |
| segment | creneau | 2π/T | 23.7 | 58 | 1297 | 1307 | 81 | 1306 | 128 | 183 |
| arc | creneau | 2π/T | 60 | 37 | 430 | 430 | 41 | 429 | 32 | 40 |
| gaussian | uniform | 4π/T | 110 | 26 | 45 | 61 | 26 | 60 | 20 | 21 |
| dipole | uniform | 4π/T | 299 | 23 | 43 | 56 | 27 | 55 | 21 | 21 |
| quarter_circle | uniform | 4π/T | 318 | 66 | 1722 | 1908 | 87 | 1907 | 67 | 81 |
| cross | uniform | 4π/T | 1.06e+03 | 69 | 1455 | 1459 | 145 | 1458 | 88 | 140 |
| disk | uniform | 4π/T | 1.95e+03 | 1 | 1 | 1 | 996 | 2057 | 1 | 1144 |
| segment | uniform | 4π/T | 18 | 128 | 2434 | 2443 | 221 | 2442 | 345 | 437 |
| arc | uniform | 4π/T | 47.8 | 83 | 776 | 778 | 106 | 777 | 85 | 98 |
| gaussian | creneau | 4π/T | 125 | 19 | 35 | 47 | 20 | 48 | 16 | 15 |
| dipole | creneau | 4π/T | 314 | 18 | 33 | 46 | 20 | 47 | 15 | 15 |
| quarter_circle | creneau | 4π/T | 359 | 50 | 1374 | 1573 | 60 | 1572 | 46 | 57 |
| cross | creneau | 4π/T | 1.08e+03 | 60 | 1416 | 1436 | 114 | 1435 | 70 | 115 |
| disk | creneau | 4π/T | 1.95e+03 | 1 | 1 | 1 | 968 | 2005 | 1 | 1254 |
| segment | creneau | 4π/T | 20.2 | 94 | 1750 | 1765 | 162 | 1764 | 232 | 305 |
| arc | creneau | 4π/T | 53.9 | 61 | 626 | 626 | 77 | 625 | 59 | 70 |

Rendu `analytic` ss=4 (défaut), quadrature de Gauss-Legendre n = 4000 (rang ≤ 4000 : les K(1e-12) proches de 4000 sont limités par la quadrature). K(seuil) = #{k : λ_k/λ_1 > seuil}. Pour le disque (invariant par rotation), tout mode au-delà du premier serait un artefact de rendu (ici λ_2/λ_1 ~ 1e-15).
