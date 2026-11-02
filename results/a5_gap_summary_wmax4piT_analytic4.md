# Expérience de gap — rendu analytic ss=4, ω_max = 4π/T = 1.2566, gap = [0.419, 0.838]

Valeurs : max de E(ω) dans le gap / moyenne dans le gap / max hors gap.

| template | régresseur | K | max gap | moy. gap | max hors gap |
|---|---|---|---|---|---|
| gaussian | trunc1e-04 | 19 | 4.93e-02 | 1.56e-02 | 1.04e-04 |
| gaussian | trunc1e-08 | 35 | 1.66e-02 | 4.40e-03 | 8.84e-09 |
| gaussian | trunc1e-12 | 47 | 4.44e-03 | 1.29e-03 | 6.44e-13 |
| gaussian | ridge1e-03 |  | 5.99e-02 | 2.20e-02 | 1.94e-03 |
| gaussian | ridge1e-06 |  | 3.23e-02 | 9.12e-03 | 2.41e-06 |
| gaussian | ridge1e-09 |  | 1.36e-02 | 3.80e-03 | 2.44e-09 |
| gaussian | P2000_trunc1e-04 |  | 5.00e-02 | 1.59e-02 | 1.23e-04 |
| segment | trunc1e-04 | 94 | 7.57e-02 | 3.18e-02 | 1.83e-04 |
| segment | trunc1e-08 | 1750 | 1.07e-02 | 8.63e-04 | 1.08e-05 |
| segment | trunc1e-12 | 1765 | 1.26e-02 | 1.79e-03 | 1.10e-05 |
| segment | ridge1e-03 |  | 7.64e-02 | 3.36e-02 | 1.70e-03 |
| segment | ridge1e-06 |  | 5.29e-02 | 1.80e-02 | 2.78e-05 |
| segment | ridge1e-09 |  | 1.28e-02 | 1.62e-03 | 1.10e-05 |
| segment | P2000_trunc1e-04 |  | 7.56e-02 | 3.18e-02 | 2.52e-04 |
| quarter_circle | trunc1e-04 | 50 | 4.07e-02 | 1.42e-02 | 5.32e-05 |
| quarter_circle | trunc1e-08 | 1374 | 3.52e-02 | 1.04e-02 | 6.24e-07 |
| quarter_circle | trunc1e-12 | 1573 | 2.77e-02 | 7.89e-03 | 6.08e-07 |
| quarter_circle | ridge1e-03 |  | 4.51e-02 | 1.70e-02 | 2.15e-03 |
| quarter_circle | ridge1e-06 |  | 3.54e-02 | 1.21e-02 | 9.68e-07 |
| quarter_circle | ridge1e-09 |  | 2.93e-02 | 8.42e-03 | 6.15e-07 |
| quarter_circle | P2000_trunc1e-04 |  | 4.05e-02 | 1.42e-02 | 5.69e-05 |
| cross | trunc1e-04 | 60 | 3.78e-02 | 1.10e-02 | 1.75e-04 |
| cross | trunc1e-08 | 1416 | 8.26e-03 | 1.73e-03 | 3.21e-06 |
| cross | trunc1e-12 | 1436 | 5.46e-02 | 1.08e-02 | 5.92e-07 |
| cross | ridge1e-03 |  | 7.23e-02 | 1.87e-02 | 6.36e-03 |
| cross | ridge1e-06 |  | 3.02e-02 | 6.97e-03 | 5.10e-06 |
| cross | ridge1e-09 |  | 1.93e-02 | 3.47e-03 | 7.11e-07 |
| cross | P2000_trunc1e-04 |  | 4.17e-02 | 1.13e-02 | 2.41e-04 |
| dipole | trunc1e-04 | 18 | 2.19e-02 | 5.44e-03 | 7.28e-05 |
| dipole | trunc1e-08 | 33 | 1.01e-02 | 2.44e-03 | 6.71e-09 |
| dipole | trunc1e-12 | 46 | 1.25e-03 | 2.02e-04 | 8.05e-13 |
| dipole | ridge1e-03 |  | 3.37e-02 | 1.06e-02 | 3.76e-03 |
| dipole | ridge1e-06 |  | 1.51e-02 | 3.69e-03 | 2.71e-06 |
| dipole | ridge1e-09 |  | 6.53e-03 | 1.54e-03 | 4.08e-09 |
| dipole | P2000_trunc1e-04 |  | 2.54e-02 | 6.88e-03 | 1.22e-03 |
