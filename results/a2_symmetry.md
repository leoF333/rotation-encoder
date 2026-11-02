# Croix vs quart de cercle vs bras seul

| template | ω_max | ρ | rendu | K(1e-4) | K(1e-12) | K_c(1e-4) | 99.9 % (non c.) | 99.9 % (c.) | 99 % (c.) |
|---|---|---|---|---|---|---|---|---|---|
| quarter_circle | 2π/T | uniform | interp ss=1 | 48 | 1509 | 56 | 42 | 49 | 23 |
| quarter_circle | 2π/T | uniform | analytic ss=4 | 40 | 1457 | 46 | 36 | 46 | 19 |
| cross | 2π/T | uniform | interp ss=1 | 50 | 1299 | 83 | 47 | 60 | 33 |
| cross | 2π/T | uniform | analytic ss=4 | 51 | 1410 | 91 | 58 | 96 | 35 |
| arm | 2π/T | uniform | interp ss=1 | 61 | 1861 | 71 | 55 | 61 | 31 |
| arm | 2π/T | uniform | analytic ss=4 | 63 | 1998 | 77 | 65 | 85 | 33 |
| quarter_circle | 2π/T | creneau | interp ss=1 | 37 | 1272 | 38 | 29 | 34 | 16 |
| quarter_circle | 2π/T | creneau | analytic ss=4 | 30 | 1109 | 33 | 25 | 32 | 13 |
| cross | 2π/T | creneau | interp ss=1 | 40 | 1184 | 59 | 36 | 44 | 24 |
| cross | 2π/T | creneau | analytic ss=4 | 41 | 1361 | 66 | 42 | 71 | 26 |
| arm | 2π/T | creneau | interp ss=1 | 44 | 1683 | 49 | 38 | 42 | 22 |
| arm | 2π/T | creneau | analytic ss=4 | 46 | 1996 | 53 | 45 | 58 | 23 |
| quarter_circle | 4π/T | uniform | interp ss=1 | 81 | 1815 | 102 | 78 | 88 | 39 |
| quarter_circle | 4π/T | uniform | analytic ss=4 | 66 | 1908 | 87 | 67 | 81 | 33 |
| cross | 4π/T | uniform | interp ss=1 | 69 | 1426 | 126 | 69 | 91 | 52 |
| cross | 4π/T | uniform | analytic ss=4 | 69 | 1459 | 146 | 88 | 138 | 54 |
| arm | 4π/T | uniform | interp ss=1 | 110 | 1990 | 135 | 105 | 113 | 56 |
| arm | 4π/T | uniform | analytic ss=4 | 113 | 1999 | 143 | 124 | 150 | 60 |
| quarter_circle | 4π/T | creneau | interp ss=1 | 59 | 1617 | 69 | 53 | 61 | 28 |
| quarter_circle | 4π/T | creneau | analytic ss=4 | 50 | 1572 | 60 | 46 | 57 | 24 |
| cross | 4π/T | creneau | interp ss=1 | 59 | 1352 | 102 | 58 | 73 | 40 |
| cross | 4π/T | creneau | analytic ss=4 | 60 | 1436 | 113 | 70 | 114 | 42 |
| arm | 4π/T | creneau | interp ss=1 | 76 | 1941 | 94 | 71 | 78 | 40 |
| arm | 4π/T | creneau | analytic ss=4 | 79 | 1998 | 99 | 84 | 105 | 42 |

## Effet du rendu (ρ uniforme, ω_max = 4π/T, n = 2000)

| template | rendu | K(1e-4) | K(1e-8) | K(1e-12) |
|---|---|---|---|---|
| gaussian | interp ss=1 | 26 | 244 | 1664 |
| gaussian | analytic ss=1 | 26 | 45 | 61 |
| gaussian | analytic ss=4 | 26 | 45 | 61 |
| gaussian | analytic ss=8 | 26 | 45 | 61 |
| quarter_circle | interp ss=1 | 81 | 579 | 1815 |
| quarter_circle | analytic ss=1 | 130 | 1269 | 1269 |
| quarter_circle | analytic ss=4 | 66 | 1461 | 1908 |
| quarter_circle | analytic ss=8 | 65 | 1068 | 1961 |
| cross | interp ss=1 | 69 | 568 | 1426 |
| cross | analytic ss=1 | 175 | 1349 | 1349 |
| cross | analytic ss=4 | 69 | 1368 | 1459 |
| cross | analytic ss=8 | 69 | 1083 | 1459 |
