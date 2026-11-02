"""aerot — rotation videos, spectra, linear encoders, replica theory and autoencoders.

Modules
-------
templates   continuous templates s0(x, y)
render      vectorised rendering of the videos phi(omega), laws of omega, quadratures (torch)
spectra     spectra of the second moment via the Gram trick, truncated pseudo-inverse, ridge
linreg      exact ridge path with P samples, Gaussian surrogates
continuum   pixel-free model: angular power spectrum and exact features
theory      replica formulas: self-consistent kappa, E_train, E_gen, E_OOD, Runge identity (numpy only)
ae          autoencoders (torch), latent metrics
fits, plotting, paths
"""
from . import theory  # noqa: F401  (pure numpy; torch-dependent modules are imported on demand)

__version__ = "1.0.0"
