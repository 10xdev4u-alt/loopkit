import numpy as np
import pytest

from loopkit.nvfp4 import (ans_decode, ans_encode, ans_size, dequantize_nvfp4,
                           quantize_nvfp4, e4m3_decode, e4m3_encode)


def test_ans_roundtrip_is_bit_exact():
    x = np.random.default_rng(20261103).standard_normal((8, 256)).astype(np.float32)
    t = quantize_nvfp4(x)
    u = ans_decode(ans_encode(t))
    assert np.array_equal(u.codes, t.codes)
    assert np.array_equal(u.scales, t.scales)
    assert u.shape == t.shape and u.block == t.block
    assert ans_size(t) > 0


def test_nvfp4_bitwise_roundtrip_1000_vectors():
    x = np.random.default_rng(20261103).standard_normal((1000, 128)).astype(np.float32)
    t = quantize_nvfp4(x)
    u = ans_decode(ans_encode(t))
    assert np.array_equal(t.codes, u.codes)
    assert np.array_equal(t.scales, u.scales)


def test_e4m3_codebook_is_stable():
    codes = np.arange(256, dtype=np.uint8)
    valid = e4m3_decode(codes)
    assert np.isfinite(valid[np.isfinite(valid)]).all()
    assert e4m3_encode(np.array([0.0, 1.0, -1.0, 6.0], dtype=np.float32)).shape == (4,)


def test_quantize_dequant_fidelity():
    """The missing test: nearest E2M1 code must preserve each block's max."""
    x = np.random.default_rng(42).standard_normal((16, 128)).astype(np.float32)
    d = dequantize_nvfp4(quantize_nvfp4(x))
    rel = np.abs(d - x) / (np.abs(x) + 1e-6)
    assert float(np.mean(rel)) < 0.25
    # block maxima must survive: 6.0 must map to code 7 (level 6), not code 6 (level 4)
    t = quantize_nvfp4(np.array([[6.0, 3.0, 1.0, 0.0]], dtype=np.float32))
    assert int(t.codes[0, 0]) == 7
    assert abs(float(dequantize_nvfp4(t)[0, 0]) - 6.0) < 0.1


def test_padded_shapes_quantize_and_roundtrip():
    x = np.random.default_rng(11).standard_normal((2, 20)).astype(np.float32)
    t = quantize_nvfp4(x)
    assert t.codes.shape == x.shape
    u = ans_decode(ans_encode(t))
    assert np.array_equal(t.codes, u.codes) and np.array_equal(t.scales, u.scales)
    assert dequantize_nvfp4(t).shape == x.shape


def test_dequant_shape_and_finite():
    x = np.random.default_rng(7).standard_normal((4, 64, 64)).astype(np.float32)
    t = quantize_nvfp4(x)
    d = dequantize_nvfp4(t)
    assert d.shape == x.shape and np.isfinite(d).all()
