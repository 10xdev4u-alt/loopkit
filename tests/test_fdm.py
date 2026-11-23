import numpy as np
from loopkit.fdm import fdm_check, fdm_dot, fused_available, reference_dot
from loopkit.nvfp4 import quantize_nvfp4


def test_fdm_dot_matches_reference():
    rng = np.random.default_rng(20261103)
    a = quantize_nvfp4(rng.normal(size=(32, 64)).astype(np.float32))
    b = quantize_nvfp4(rng.normal(size=(64, 32)).astype(np.float32))
    ref = reference_dot(a, b)
    r = fdm_dot(a, b)
    assert r.max_abs_err_vs_ref == 0.0
    assert np.array_equal(r.y, ref)


def test_fused_available_is_honest():
    assert fused_available() is False


def test_fdm_check_smoke():
    r = fdm_check()
    assert r.path in ("fused-staging", "dequant-fp16")
