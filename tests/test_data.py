"""Rendering, spectra and the angular power spectrum (requires torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.special import ive  # noqa: E402

from aerot import continuum, render, spectra, templates  # noqa: E402


def test_disk_is_rotation_invariant():
    """Negative control: a centred disk gives a rank-one second moment."""
    om, w = render.quadrature("uniform", 2 * np.pi / 10, 200)
    Phi = render.make_videos(om, "disk", d=20, T=6).numpy()
    S = spectra.GramSpectrum(Phi, w, om)
    assert S.lam[1] / S.lam[0] < 1e-10


def test_video_shape_and_rotation():
    """Frame t of the analytic rendering equals the template rotated by omega t."""
    om = np.array([0.3])
    vid = render.make_videos(om, "gaussian", d=28, T=5, flat=False)[0]
    for t in range(5):
        img = render.raster("gaussian", 28, render.DEFAULT_SS, angle=0.3 * t)
        assert torch.allclose(vid[t], img, atol=1e-12)


def test_gaussian_angular_power_is_bessel():
    """E_m proportional to e^{-z0} I_m(z0), z0 = r0^2 / (2 sigma^2) (Jacobi-Anger)."""
    p = templates.template_params("gaussian", 28)
    Ek = continuum.angular_power("gaussian", p, d=28, r_max=40, nr=600, nth=1024, oversample=2)
    z0 = p["r0"] ** 2 / (2 * p["sigma"] ** 2)
    m = np.arange(0, 12)
    ratio = Ek[m] / ive(m, z0)
    assert np.allclose(ratio / ratio[0], 1.0, rtol=2e-3)


def test_cross_selection_rule():
    """Four-fold symmetry: angular harmonics m not multiple of 4 vanish."""
    Ek = continuum.angular_power("cross", d=28, nr=200, nth=512, oversample=2)
    m = np.arange(1, 24)
    off = Ek[m][m % 4 != 0]
    on = Ek[m][m % 4 == 0]
    assert off.max() < 1e-6 * on.max()


def test_truncated_pinv_matches_theory_reference():
    """Population E_OOD of the truncated pseudo-inverse: GramSpectrum vs aerot.theory on the same moments."""
    from aerot import theory as th
    wmax = 2 * np.pi / 10
    om, w = render.quadrature("creneau", wmax, 400)
    Phi = render.make_videos(om, "gaussian", d=16, T=6).numpy()
    S = spectra.GramSpectrum(Phi, w, om)
    omt, wt = render.quadrature("gap", wmax, 200)
    Phit = render.make_videos(omt, "gaussian", d=16, T=6).numpy()
    K = S.n_above(1e-10)
    tq = S.test_quantities(Phit, wt, omt, K)
    W = S.weights_trunc(1e-4)
    direct = float(wt @ (omt - Phit @ W) ** 2)
    ref = th.ood_error_truncated(S.lam[:K], S.c_proj[:K], tq["test_cov_in_train_basis"], tq["test_c_proj"], tq["test_omega2"], tau=1e-4)
    assert direct == pytest.approx(ref, rel=1e-6)
