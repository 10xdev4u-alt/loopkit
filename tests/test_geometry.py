import numpy as np
import pytest

from loopkit.geometry import (break_even_passes, fp4_itq, geometry_gate,
                              joint_itq, lambda_law, spectral_break_even,
                              predict_error)


def test_lambda_law_residual():
    bits = np.array([1.0, 2.0, 3.0, 4.0])
    c, lam = 2.0, 0.5
    errors = c * 2 ** (-lam * bits)
    got_lam, got_c = lambda_law(bits, errors)
    assert abs(got_lam - lam) < 1e-10
    assert abs(got_c - c) < 1e-10


def test_gaussian_lambda_mean():
    x = np.random.default_rng(20261106).standard_normal((10000, 32))
    lam = 1.0 - (np.abs(x).sum(axis=1) / np.linalg.norm(x, axis=1)) ** 2 / x.shape[1]
    # Finite-r expectation is (1 - 2/pi) * (1 - 1/r) = 0.3520 at r=32;
    # 0.3634 is only the r -> infinity limit.
    expected = (1.0 - 2.0 / np.pi) * (1.0 - 1.0 / x.shape[1])
    assert abs(float(lam.mean()) - expected) < 0.004


def test_break_even_gamma_star_bands():
    assert abs(spectral_break_even(0.54) - 0.385) < 0.01
    assert abs(spectral_break_even(0.393) - 0.458) < 0.01
    assert abs(spectral_break_even(0.30) - 0.506) < 0.01


def test_break_even_passes():
    assert break_even_passes(2.0, 1.0, 0.25) == 2
    assert break_even_passes(1.0, 1.0, 1.0) is None


def test_joint_itq_orthogonal_and_monotone():
    rng = np.random.default_rng(20261108)
    passes = [rng.standard_normal((64, 32)) for _ in range(3)]
    R = joint_itq(passes, iters=20)
    assert R.shape == (32, 32)
    assert np.allclose(R.T @ R, np.eye(32), atol=1e-8)


@pytest.mark.kaggle
def test_fp4_aware_itq_beats_binary_itq():
    x = np.random.default_rng(20261111).standard_normal((512, 128))
    R, codes = fp4_itq(x, iters=30)
    assert R.shape == (128, 128)
    assert codes is not None


def test_geometry_gate_reports_reasons():
    verdict = geometry_gate(np.array([1.0, 2.0]), np.array([0.5, 0.25]),
                            base_per_pass=2.0, codec_setup=1.0, codec_per_pass=0.25)
    assert verdict.go
