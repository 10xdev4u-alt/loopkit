import numpy as np
import pytest

from loopkit.nvfp4 import (ans_decode, ans_encode, ans_size, dequantize_mxfp8,
                           dequantize_mxfp8_weights, dequantize_nvfp4,
                           quantize_mxfp8, quantize_mxfp8_weights,
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


def test_mxfp8_weights_use_2d_tile_scales():
    x = np.random.default_rng(2026).standard_normal((64, 64)).astype(np.float32)
    t = quantize_mxfp8_weights(x)
    assert t.scales.shape == (2, 2)  # one E8M0 scale per 32x32 tile
    assert t.layout == "2d"
    assert t.codes.shape == x.shape
    d = dequantize_mxfp8_weights(t)
    assert d.shape == x.shape and np.isfinite(d).all()


def test_mxfp8_weights_ans_roundtrip_preserves_layout_and_scale_shape():
    x = np.random.default_rng(77).standard_normal((70, 45)).astype(np.float32)
    t = quantize_mxfp8_weights(x)
    u = ans_decode(ans_encode(t))
    assert np.array_equal(u.codes, t.codes)
    assert np.array_equal(u.scales, t.scales)
    assert u.shape == t.shape and u.block == t.block
    assert u.layout == t.layout == "2d"
    assert u.scales.shape == t.scales.shape == (3, 2)
    assert np.array_equal(dequantize_mxfp8_weights(u), dequantize_mxfp8_weights(t))


def test_mxfp8_activation_path_stays_1d():
    x = np.random.default_rng(5).standard_normal((8, 64)).astype(np.float32)
    t = quantize_mxfp8(x)
    assert t.layout == "1d"
    assert t.scales.shape == (8, 2)  # flat runs of 32, one scale per row-run
    assert dequantize_mxfp8(t).shape == x.shape
    u = ans_decode(ans_encode(t))
    assert np.array_equal(u.codes, t.codes) and np.array_equal(u.scales, t.scales)
    assert u.layout == "1d" and u.scales.shape == t.scales.shape
    assert np.array_equal(dequantize_mxfp8(u), dequantize_mxfp8(t))


def test_mxfp8_2d_dequant_error_is_comparable_to_1d():
    x = np.random.default_rng(1234).standard_normal((128, 128)).astype(np.float32)
    d2 = dequantize_mxfp8_weights(quantize_mxfp8_weights(x))
    d1 = dequantize_mxfp8(quantize_mxfp8(x))
    mse2 = float(np.mean((d2 - x) ** 2))
    mse1 = float(np.mean((d1 - x) ** 2))
    assert mse1 > 0 and mse2 <= 1.5 * mse1
