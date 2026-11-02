# Expérience de gap — rendu analytic ss=4, ω_max = 2π/T = 0.6283, gap = [0.209, 0.419]

Valeurs : max de E(ω) dans le gap / moyenne dans le gap / max hors gap.

| template | régresseur | K | max gap | moy. gap | max hors gap |
|---|---|---|---|---|---|
| gaussian | trunc1e-04 | 12 | 7.48e-04 | 2.79e-04 | 1.64e-05 |
| gaussian | trunc1e-08 | 21 | 5.67e-06 | 2.14e-06 | 2.69e-09 |
| gaussian | trunc1e-12 | 28 | 3.53e-06 | 7.97e-07 | 4.04e-13 |
| gaussian | ridge1e-03 |  | 9.41e-04 | 4.38e-04 | 4.27e-04 |
| gaussian | ridge1e-06 |  | 8.82e-05 | 3.08e-05 | 3.98e-07 |
| gaussian | ridge1e-09 |  | 4.32e-06 | 1.25e-06 | 5.34e-10 |
| gaussian | P2000_trunc1e-04 |  | 7.55e-04 | 2.81e-04 | 2.00e-05 |
| segment | trunc1e-04 | 58 | 7.81e-03 | 4.11e-03 | 1.51e-05 |
| segment | trunc1e-08 | 1297 | 3.54e-03 | 1.62e-03 | 2.39e-06 |
| segment | trunc1e-12 | 1307 | 3.96e-03 | 1.31e-03 | 3.58e-06 |
| segment | ridge1e-03 |  | 8.51e-03 | 4.71e-03 | 2.89e-04 |
| segment | ridge1e-06 |  | 5.19e-03 | 2.59e-03 | 2.22e-06 |
| segment | ridge1e-09 |  | 3.84e-03 | 1.31e-03 | 3.16e-06 |
| segment | P2000_trunc1e-04 |  | 7.92e-03 | 4.19e-03 | 1.46e-05 |
| quarter_circle | trunc1e-04 | 30 | 3.32e-04 | 1.54e-04 | 1.28e-05 |
| quarter_circle | trunc1e-08 | 1004 | 4.83e-04 | 1.26e-04 | 5.82e-07 |
| quarter_circle | trunc1e-12 | 1110 | 6.92e-04 | 2.30e-04 | 5.75e-07 |
| quarter_circle | ridge1e-03 |  | 4.77e-04 | 2.55e-04 | 3.81e-04 |
| quarter_circle | ridge1e-06 |  | 2.50e-04 | 1.08e-04 | 1.38e-06 |
| quarter_circle | ridge1e-09 |  | 5.61e-04 | 1.85e-04 | 5.79e-07 |
| quarter_circle | P2000_trunc1e-04 |  | 3.43e-04 | 1.61e-04 | 1.72e-05 |
| cross | trunc1e-04 | 41 | 6.54e-03 | 2.24e-03 | 6.90e-05 |
| cross | trunc1e-08 | 1298 | 3.26e-04 | 1.21e-04 | 3.65e-07 |
| cross | trunc1e-12 | 1361 | 3.72e-03 | 9.70e-04 | 3.63e-07 |
| cross | ridge1e-03 |  | 7.00e-03 | 2.58e-03 | 9.63e-04 |
| cross | ridge1e-06 |  | 4.50e-03 | 1.63e-03 | 6.67e-07 |
| cross | ridge1e-09 |  | 6.22e-04 | 1.39e-04 | 3.63e-07 |
| cross | P2000_trunc1e-04 |  | 6.63e-03 | 2.27e-03 | 8.12e-05 |
| dipole | trunc1e-04 | 12 | 1.23e-03 | 3.82e-04 | 1.71e-05 |
| dipole | trunc1e-08 | 20 | 1.76e-05 | 4.15e-06 | 4.35e-09 |
| dipole | trunc1e-12 | 28 | 1.24e-06 | 2.76e-07 | 9.53e-14 |
| dipole | ridge1e-03 |  | 2.35e-03 | 7.84e-04 | 7.76e-04 |
| dipole | ridge1e-06 |  | 1.65e-05 | 5.31e-06 | 6.47e-07 |
| dipole | ridge1e-09 |  | 7.38e-06 | 2.01e-06 | 8.95e-10 |
| dipole | P2000_trunc1e-04 |  | 1.30e-03 | 4.00e-04 | 2.12e-05 |
