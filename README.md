# Encoding a rotation: replica theory and out-of-distribution error of a linear encoder

**How does a linear encoder learn the angular velocity of a rotating object, and can it extrapolate to velocities it has never seen?**
Theory (replica method, exact linear-algebra identities), numerics (spectra, ridge regression, autoencoders) and an independent re-verification, for a research project in statistical physics of learning (École Polytechnique / ENS, supervised by Rémi Monasson, 2025–2026).

<p align="center">
<img src="figures/a5_gap_vs_memoir.png" width="100%" alt="Pointwise error inside the gap for four templates">
</p>

*Error $E(\omega)=(\omega-W\cdot\phi(\omega))^2$ of a linear encoder trained on $[0,\omega_{\max}/3]\cup[2\omega_{\max}/3,\omega_{\max}]$ and tested everywhere. A smooth object (Gaussian blob) extrapolates into the gap at $10^{-6}$; objects with sharp edges fail by three orders of magnitude.*

## The problem

A template $s_0$ ($28\times28$ pixels) rotates by $\omega t$ in each of $T=10$ frames. The flattened video $\phi(\omega)\in\mathbb R^{7840}$ lies on a one-dimensional manifold. A linear encoder $\hat\omega=W\cdot\phi$ is trained by ridge regression on $P$ videos,

$$H(W)=\tfrac12\sum_{\mu=1}^P\big(\omega_\mu-W\cdot\phi(\omega_\mu)\big)^2+\tfrac{\lambda_r}{2}\|W\|^2 ,$$

and evaluated in distribution ($E_{\rm gen}$) and inside a **gap** of velocities absent from training ($E_{\rm OOD}$).

## Main results

| | Result | Status |
|---|---|---|
| 1 | In the continuum limit the data kernel is stationary in $\omega$ and fixed by the template's **angular power spectrum** $S_m$: $S_m=\pi\sigma^2e^{-z_0}I_m(z_0)$ for a Gaussian blob ($z_0=r_0^2/2\sigma^2$), $S_m\sim m^{-2}$ for a sharp edge, $S_m=0$ unless $4\mid m$ for a cross. | proved |
| 2 | **Replica theory with ridge**: a single self-consistent spectral scale $\kappa\,(1-{\rm df}(\kappa)/P)=\lambda_r/P$ fixes $E_{\rm train}$, $E_{\rm gen}$ and $E_{\rm OOD}$ for every $\alpha=P/N$; the memoir's ridgeless calculation is recovered, corrected (two errata), and its unexplained order parameter $q_0^*$ identified as the generalisation error. | replica (exact for Gaussian covariates) |
| 3 | $E_{\rm OOD}=E_{\rm te}(\bar W_\kappa)+\frac{E_{\rm gen}}{P}\sum_k(\Sigma_{\rm te})_{kk}\frac{\lambda_k}{(\lambda_k+\kappa)^2}$ predicts the gap error of the **real videos** with a median deviation of **0.8 %** ($P\ge320$, 64 settings; 1.0 % in an independent re-implementation). | numerical validation |
| 4 | **Runge identity** (exact, no Gaussian assumption, $P=\infty$): $E_{\rm OOD}=E^{\min}_{\rm te}+\sum_j\frac{(\mu_j+2)^2}{\mu_j}\langle\varepsilon_u,g_j\rangle_{\rm tr}^2$ — the gap error is the misspecification residual projected on the directions concentrated in the gap (generalised eigenvalues $\mu_j$ of $(\Sigma_{\rm te},\Sigma_{\rm tr})$, a discrete Slepian spectrum) and amplified by their concentration. Without misspecification there is no covariate-shift error. | proved |
| 5 | The smooth / sharp-edge dichotomy is a **competition between approximation rate and amplification**: super-exponential residual decay wins for smooth templates, algebraic decay loses for edges. It only appears for a fine spectral cutoff ($\lambda_k/\lambda_1\gtrsim10^{-8}$) and $\omega_{\max}=2\pi/T$; an off-centre Gaussian blob extrapolates as badly as an edge (the error is a universal function of $r_0/\sigma$). | proved (continuum) + numerics |
| 6 | On a 1-D manifold, sampling holes of size $\sim\omega_{\max}\ln P/P$ act as mini-gaps and break the Gaussian equivalence by $\times10$–$250$ when $P$ is comparable to the number of learned modes; stratified sampling removes most of the excess. | numerics |
| 7 | In non-linear autoencoders (81 trainings, 3 templates) the latent stays monotone in $\omega$ across the gap, but no decoder reconstructs the gap frames: the out-of-distribution difficulty sits in generation, not inference. | numerics |

Full write-up: [`docs/SYNTHESIS.md`](docs/SYNTHESIS.md) (overview), [`theory/derivations/`](theory/derivations) (step-by-step derivations 01–05), [`docs/REPORT_numerics.md`](docs/REPORT_numerics.md), [`docs/REPORT_theory.md`](docs/REPORT_theory.md), [`docs/REPORT_verification.md`](docs/REPORT_verification.md), [`docs/AUDIT_2026-10-15.md`](docs/AUDIT_2026-10-15.md) (status of every claim: proved / replica / numerical).

<p align="center">
<img src="theory/figures/t4_gap_runge.png" width="100%" alt="Gap error versus cutoff, Runge decomposition, ridge">
</p>

## Repository layout

```
src/aerot/          installable package
  templates.py      continuous templates s0(x, y)
  render.py         vectorised rendering of phi(omega) (torch), laws of omega, quadratures
  spectra.py        spectrum of <phi phi^T> by the Gram trick, truncated pinv, ridge, test moments
  linreg.py         exact ridge path at finite P, Gaussian surrogates
  continuum.py      pixel-free model: angular power spectrum S_m, exact features
  theory.py         replica formulas (numpy only): kappa, E_train, E_gen, E_OOD, Runge decomposition
  ae.py             autoencoders (torch) and latent metrics
scripts/            one script per experiment (A1–A6 reproduction, B1–B3 extensions); see scripts/README.md
theory/             derivations (Markdown + LaTeX), check scripts (sympy, Monte-Carlo), figures
verification/       independent re-implementation of the key results, and the Oct. 2026 audit
docs/               synthesis, reports, conventions, the original memoir (PDF)
figures/, results/  generated figures (with their parameters as .json) and curated results
tests/              pytest: replica formulas vs. Gaussian simulations, exact identities, rendering checks
```

## Install and run

```bash
conda env create -f environment.yml && conda activate aerot      # or: pip install -e ".[full,test]"
pytest                                                           # ~1 min, CPU
python scripts/a2_export_spectra.py                              # population spectra (prerequisite, ~5 min)
python scripts/a5_gap.py --wmax 2                                # the gap experiment
python theory/checks/test_theory.py                              # longer theory checks
```

Every figure in `figures/` has one command in [`scripts/README.md`](scripts/README.md). Scripts are idempotent and re-use `results/`. The GPU code runs on Apple MPS, CUDA or CPU (`aerot.render.get_device`). Large regenerable caches (`data/spectra/*.npz`, `results/b1_theory_cache/`) are git-ignored.

Minimal use of the theory module:

```python
import numpy as np
from aerot import theory as th

lam = np.arange(1, 201) ** -1.3              # spectrum of <phi phi^T>
c = 0.3 * lam / np.sqrt(np.arange(1, 201))  # projections of <omega phi>
omega2 = 0.03 + np.sum(c**2 / lam)          # <omega^2>: residual 0.03 + explained power
th.gen_error(lam, c, omega2, P=300, lam_r=0.1), th.train_error(lam, c, omega2, P=300, lam_r=0.1)
```

## Conventions

Non-centred second moments $\Sigma=\langle\phi\phi^\top\rangle$, $C=\langle\omega\phi\rangle$; errors per sample without the factor ½; the ridge compares to the eigenvalues through $\lambda_r/P$. See [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

## Provenance and honesty note

The analytical and numerical work extends a research memoir written in 2025–2026 (`docs/memoire_PRL_2025-2026.pdf`). The reconstruction and extension phase (September–October 2026) was carried out with AI coding agents (Claude, Anthropic) directed by the author; every structural result was then re-derived by hand or re-implemented in an independent code (`verification/`), and the status of each claim — proved, replica-level, or numerical — is tracked in `docs/AUDIT_2026-10-15.md`. Code comments and derivations are in French.

## References

- Hastie, Montanari, Rosset, Tibshirani, *Surprises in high-dimensional ridgeless least squares interpolation*, Ann. Statist. (2022)
- Bordelon, Canatar, Pehlevan, *Spectrum dependent learning curves in kernel regression and wide neural networks*, ICML (2020)
- Canatar, Bordelon, Pehlevan, *Out-of-distribution generalization in kernel regression*, NeurIPS (2021)
- Patil, Du, Tibshirani, *Optimal ridge regularization for out-of-distribution prediction*, ICML (2024)
- Bach, *High-dimensional analysis of double descent for linear regression with random projections*, SIAM J. Math. Data Sci. (2024)

## License

MIT.
