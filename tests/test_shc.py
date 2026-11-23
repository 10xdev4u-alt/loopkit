import numpy as np

from loopkit.shc import HealthCard, health_card, health_report, recover_weights


def test_recover_weights_matches_known_dense_matrix():
    rng = np.random.default_rng(0)
    n, d_in, d_out = 200, 12, 5
    x = rng.standard_normal((n, d_in)) * 10.0
    w = rng.standard_normal((d_in, d_out))
    got = recover_weights(x, y=x @ w)
    assert got.shape == (d_in, d_out)
    assert got.dtype == np.float64
    assert np.allclose(got, w, rtol=0.0, atol=1e-6)


def test_health_card_flags_outlier_dim_and_not_dense_layer():
    rng = np.random.default_rng(1)
    n, d_in, d_out = 512, 2304, 32

    x_dense = rng.standard_normal((n, d_in))
    w_dense = rng.standard_normal((d_in, d_out))
    dense = health_card("dense", x_dense, x_dense @ w_dense)
    assert dense.n_samples == n
    assert not dense.flagged
    assert dense.lambda_max < 0.6
    assert dense.kurtosis < 8.0

    # The activation outlier at dim 2027 makes that direction observable even
    # with n_samples < d_in; the 16x weight spike then dominates the recovery.
    x_spiky = rng.standard_normal((n, d_in))
    x_spiky[:, 2027] *= 20.0
    w_spiky = rng.standard_normal((d_in, d_out))
    w_spiky[2027] *= 16.0
    spiky = health_card("spiky", x_spiky, x_spiky @ w_spiky)
    assert spiky.flagged
    assert spiky.lambda_max > 0.6 or spiky.kurtosis > 8.0


def test_gamma_fit_recovers_power_law_exponent():
    rng = np.random.default_rng(2)
    d, n = 256, 512
    u, _ = np.linalg.qr(rng.standard_normal((d, d)))
    v, _ = np.linalg.qr(rng.standard_normal((d, d)))
    k = np.arange(1, d + 1, dtype=np.float64)
    w = (u * k**-0.27) @ v.T
    x = rng.standard_normal((n, d)) * 10.0
    card = health_card("power-law", x, x @ w)
    assert abs(card.gamma_hat - 0.27) < 0.05


def test_health_report_returns_one_card_per_key():
    rng = np.random.default_rng(3)
    samples = {}
    for name in ("layer.00", "layer.11", "layer.22"):
        x = rng.standard_normal((64, 8))
        samples[name] = (x, x @ rng.standard_normal((8, 4)))
    report = health_report(samples)
    assert set(report) == set(samples)
    for name, card in report.items():
        assert isinstance(card, HealthCard)
        assert card.layer == name
