"""Replica formulas (aerot.theory) against direct simulations on Gaussian data and exact identities.

Fast (< 1 min, CPU, numpy only). The heavier original check scripts live in theory/checks/.
"""
import numpy as np
import pytest

from aerot import theory as th


def _ridge_sim(rng, lam, theta, s2, P, lam_r, n_trial, Ste=None, Cte=None, o2te=None):
    """Mean (E_train, E_gen[, E_test]) of the ridge estimator over n_trial Gaussian training sets."""
    N = lam.size
    acc = np.zeros(3)
    for _ in range(n_trial):
        X = rng.standard_normal((P, N)) * np.sqrt(lam)
        y = X @ theta + rng.standard_normal(P) * np.sqrt(s2)
        W = np.linalg.solve(X.T @ X + lam_r * np.eye(N), X.T @ y)
        d = W - theta
        acc[0] += np.mean((y - X @ W) ** 2)
        acc[1] += s2 + d @ (lam * d)
        if Ste is not None:
            acc[2] += o2te - 2 * Cte @ W + W @ Ste @ W
    return acc / n_trial


@pytest.fixture
def spectrum(rng):
    N = 120
    lam = np.arange(1, N + 1) ** -1.3
    theta = rng.choice([-1.0, 1.0], N) * np.sqrt(0.4 * np.arange(1, N + 1) ** -1.2 / lam)
    s2 = 0.03
    c = lam * theta
    omega2 = s2 + np.sum(c**2 / lam)
    return lam, theta, s2, c, omega2


def test_kappa_equation_is_satisfied(spectrum):
    lam, _, _, _, _ = spectrum
    for P, lam_r in [(60, 0.5), (300, 1e-3), (1000, 10.0)]:
        kap = th.solve_kappa(lam, P, lam_r)
        assert kap > 0
        assert abs(kap * (1 - th.df_kappa(lam, kap) / P) - lam_r / P) < 1e-10 * max(1.0, lam_r / P)


def test_residual_variance(spectrum):
    lam, _, s2, c, omega2 = spectrum
    assert th.residual_variance(lam, c, omega2) == pytest.approx(s2, rel=1e-10)


@pytest.mark.parametrize("P,lam_r", [(60, 1.0), (300, 0.1), (400, 10.0)])
def test_train_and_gen_error_match_simulation(rng, spectrum, P, lam_r):
    lam, theta, s2, c, omega2 = spectrum
    sim = _ridge_sim(rng, lam, theta, s2, P, lam_r, n_trial=150)
    assert th.train_error(lam, c, omega2, P, lam_r) == pytest.approx(sim[0], rel=0.05)
    assert th.gen_error(lam, c, omega2, P, lam_r) == pytest.approx(sim[1], rel=0.05)


def test_train_gen_ratio_identity(spectrum):
    lam, _, _, c, omega2 = spectrum
    P, lam_r = 200, 0.3
    op = th.order_parameters(lam, c, omega2, P, lam_r)
    assert op["Etrain"] == pytest.approx((1 - op["df"] / P) ** 2 * op["Egen"], rel=1e-9)


def test_ridgeless_limit_classical_least_squares(spectrum):
    """P > N, lambda_r -> 0 : E_train = s2 (1 - N/P), E_gen = s2 / (1 - N/P)."""
    lam, _, s2, c, omega2 = spectrum
    N = lam.size
    P = 4 * N
    alpha = P / N
    assert th.train_error(lam, c, omega2, P, 1e-12) == pytest.approx(s2 * (1 - 1 / alpha), rel=1e-4)
    assert th.gen_error(lam, c, omega2, P, 1e-12) == pytest.approx(s2 * alpha / (alpha - 1), rel=1e-4)


def test_ood_error_matches_simulation(rng, spectrum):
    lam, theta, s2, c, omega2 = spectrum
    N = lam.size
    B = np.diag(np.sqrt(lam)) + 0.05 * rng.standard_normal((N, N)) * (np.arange(N)[:, None] < 40)
    Ste = B @ B.T                                   # non-diagonal test covariance in the training basis
    theta_te = theta + 0.3 * rng.standard_normal(N) / np.sqrt(np.arange(1, N + 1))
    Cte = Ste @ theta_te
    o2te = theta_te @ Ste @ theta_te + 0.01
    for P, lam_r in [(300, 0.3), (1000, 1e-3)]:
        sim = _ridge_sim(rng, lam, theta, s2, P, lam_r, n_trial=150, Ste=Ste, Cte=Cte, o2te=o2te)
        assert th.ood_error(lam, c, omega2, Ste, Cte, o2te, P, lam_r) == pytest.approx(sim[2], rel=0.05)


def test_ood_error_reduces_to_gen_error_in_distribution(spectrum):
    lam, _, _, c, omega2 = spectrum
    P, lam_r = 150, 0.2
    E = th.ood_error(lam, c, omega2, np.diag(lam), c, omega2, P, lam_r)
    assert E == pytest.approx(th.gen_error(lam, c, omega2, P, lam_r), rel=1e-9)


def test_runge_identity_on_a_1d_manifold():
    """Exact identity E_OOD = E_min + sum_j (mu_j+2)^2/mu_j e_j^2 on cosine features, train = 2/3 of [0,1], gap = middle third."""
    from numpy.polynomial.legendre import leggauss
    K = 8
    s = np.exp(-0.3 * np.arange(K))
    feat = lambda x: np.cos(np.pi * np.outer(x, np.arange(K))) * s  # noqa: E731

    def moments(segs):
        F, Y, Wt = [], [], []
        tot = sum(b - a for a, b in segs)
        for a, b in segs:
            x, w = leggauss(300)
            x = 0.5 * (b - a) * x + 0.5 * (b + a)
            F.append(feat(x)); Y.append(x); Wt.append(0.5 * (b - a) * w / tot)
        F, Y, Wt = np.vstack(F), np.concatenate(Y), np.concatenate(Wt)
        return (F * Wt[:, None]).T @ F, (F * Wt[:, None]).T @ Y, float(Wt @ Y**2)

    Str, Ctr, o2tr = moments([(0, 1 / 3), (2 / 3, 1)])
    Ste, Cte, o2te = moments([(1 / 3, 2 / 3)])
    lam, U = np.linalg.eigh(Str)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    r = th.runge_decomposition(lam, U.T @ Ctr, o2tr, U.T @ Ste @ U, U.T @ Cte, o2te, K=K)
    assert r["Eood"] == pytest.approx(r["Emin"] + r["shift"], rel=1e-8)
    # the memoir's truncated pseudo-inverse at tau -> 0 gives the same number
    assert th.ood_error_truncated(lam, U.T @ Ctr, U.T @ Ste @ U, U.T @ Cte, o2te, tau=1e-14) == pytest.approx(r["Eood"], rel=1e-8)


def test_ridgeless_saddle_point_sign():
    """The corrected m* makes the free-energy combination equal to sigma2_res; the memoir's sign does not."""
    Jcc, Jpc, Jpp, wbar, s2 = 0.3, 0.2, 0.5, 0.7, 0.1
    corrected = th.memoire_ridgeless(Jcc, Jpc, Jpp, wbar, s2, alpha=3.0, corrected=True)
    original = th.memoire_ridgeless(Jcc, Jpc, Jpp, wbar, s2, alpha=3.0, corrected=False)
    sigma2_res = s2 - Jcc + (wbar - Jpc) ** 2 / (1 + Jpp)
    assert corrected["X"] == pytest.approx(sigma2_res, rel=1e-12)
    assert original["X"] != pytest.approx(sigma2_res, rel=1e-6)
    assert corrected["Egen"] == pytest.approx(3.0 / 2.0 * sigma2_res, rel=1e-12)
