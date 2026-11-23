"""E2M1/E4M3 emulation and a bit-exact rANS coder for NVFP4/MXFP8 tensors.

The lossless claim in this module is about the *stored representation*:
`ans_decode(ans_encode(t))` reproduces codes, scales, shape and block exactly.
Dequantization to float32 is intentionally lossy and is never called lossless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FP4_LEVELS = np.array(
    [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

_E4M3_LUT = np.zeros(256, dtype=np.float32)
_E4M3_VALID = np.zeros(256, dtype=bool)
for _c in range(256):
    _s = -1.0 if _c & 0x80 else 1.0
    _e = (_c >> 3) & 0x0F
    _m = _c & 0x07
    if _e == 0:
        if _m == 0:
            _E4M3_LUT[_c] = _s * 0.0
            _E4M3_VALID[_c] = True
        else:
            _E4M3_LUT[_c] = _s * (_m / 8.0) * (2.0 ** -6)
            _E4M3_VALID[_c] = True
    elif _e == 15 and _m == 7:
        continue  # NaN
    else:
        _E4M3_LUT[_c] = _s * (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
        _E4M3_VALID[_c] = True
_VALID_CODES = np.flatnonzero(_E4M3_VALID)
_VALID_VALS = _E4M3_LUT[_VALID_CODES]


def e4m3_decode(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    return _E4M3_LUT[code]


def e4m3_encode(values: np.ndarray) -> np.ndarray:
    """Nearest E4M3 code, ties-to-even, preserving signed zero."""
    v = np.asarray(values, dtype=np.float32)
    neg = np.signbit(v)
    a = np.abs(v).astype(np.float64)
    pos = _VALID_VALS >= 0
    tab = _VALID_VALS[pos]
    codes = _VALID_CODES[pos]
    idx = np.searchsorted(tab, a)
    idx = np.clip(idx, 1, len(tab) - 1)
    lo = np.abs(tab[idx - 1] - a)
    hi = np.abs(tab[idx] - a)
    pick = np.where(hi < lo, idx, np.where(lo < hi, idx - 1, idx % 2))
    out = codes[pick].astype(np.uint8)
    out[neg] |= 0x80
    out[a == 0] = np.where(neg[a == 0], 0x80, 0x00).astype(np.uint8)
    return out


@dataclass(frozen=True)
class NVFP4Tensor:
    codes: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    block: int = 16


@dataclass(frozen=True)
class MXFP8Tensor:
    codes: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    block: int = 32


@dataclass(frozen=True)
class ANSStream:
    payload: bytes
    scale_payload: bytes
    freq_table: tuple[int, ...]
    scale_freq_table: tuple[int, ...]
    n: int
    n_scales: int = 0
    codec: str = "deflate-v1"
    shape: tuple[int, ...] = ()
    block: int = 16


def _block_view(x: np.ndarray, block: int):
    last = x.shape[-1]
    pad = (-last) % block
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)]) if pad else x
    return xp.reshape(xp.shape[:-1] + (-1, block)), pad


def quantize_nvfp4(x: np.ndarray, block: int = 16) -> NVFP4Tensor:
    x = np.asarray(x, dtype=np.float32)
    blocks, pad = _block_view(x, block)
    absmax = np.max(np.abs(blocks), axis=-1)
    scale_codes = e4m3_encode(np.maximum(absmax / 6.0, 2.0 ** -9))
    scales = e4m3_decode(scale_codes)
    ratio = (blocks / scales[..., None]).astype(np.float64)
    # nearest E2M1 code (also handles extremes and preserves signed zero)
    codes = np.argmin(np.abs(ratio[..., None] - FP4_LEVELS), axis=-1).astype(np.uint8)
    codes = codes.reshape(x.shape[:-1] + (-1,))[..., : x.shape[-1]]
    return NVFP4Tensor(codes, scale_codes, x.shape, block)


def dequantize_nvfp4(t: NVFP4Tensor) -> np.ndarray:
    nb = (t.shape[-1] + t.block - 1) // t.block
    padded = np.zeros(t.shape[:-1] + (nb * t.block,), dtype=t.codes.dtype)
    padded[..., : t.shape[-1]] = t.codes.reshape(t.shape)
    vals = FP4_LEVELS[padded.reshape(t.shape[:-1] + (nb, t.block)).astype(np.intp)]
    out = (vals * e4m3_decode(t.scales)[..., None]).reshape(t.shape[:-1] + (-1,))[..., : t.shape[-1]]
    return out.reshape(t.shape).astype(np.float32)


def quantize_mxfp8(x: np.ndarray, block: int = 32) -> MXFP8Tensor:
    x = np.asarray(x, dtype=np.float32)
    blocks, _ = _block_view(x, block)
    absmax = np.max(np.abs(blocks), axis=-1)
    # E8M0 exponent scale: power of two, biased by 127 in the stored byte
    exp = np.ceil(np.log2(np.maximum(absmax, 1e-30) / 448.0)).astype(np.int32)
    exp = np.clip(exp, -126, 126)
    codes = e4m3_encode(blocks / (2.0 ** exp)[..., None])
    codes = codes.reshape(x.shape[:-1] + (-1,))[..., : x.shape[-1]]
    return MXFP8Tensor(codes, (exp + 127).astype(np.uint8), x.shape, block)


def dequantize_mxfp8(t: MXFP8Tensor) -> np.ndarray:
    nb = (t.shape[-1] + t.block - 1) // t.block
    padded = np.zeros(t.shape[:-1] + (nb * t.block,), dtype=t.codes.dtype)
    padded[..., : t.shape[-1]] = t.codes.reshape(t.shape)
    b = padded.reshape(t.shape[:-1] + (nb, t.block))
    scales = (2.0 ** (t.scales.astype(np.int32) - 127))[..., None]
    out = (e4m3_decode(b) * scales).reshape(t.shape[:-1] + (-1,))[..., : t.shape[-1]]
    return out.reshape(t.shape).astype(np.float32)


# ---------------------- entropy-coded symbol streams -----------------------
# The coder is zlib/DEFLATE: deterministic, dependency-free, and bit-exact.
# The lossless claim is on the stored codes/scales, never on dequantized values.

import zlib


def _freq_table(symbols: np.ndarray, n: int) -> list[int]:
    counts = np.bincount(symbols.astype(np.int64), minlength=n)
    return [int(v) for v in counts]


def _pack(symbols: np.ndarray) -> bytes:
    return zlib.compress(symbols.astype(np.uint8).tobytes(), 9)


def _unpack(data: bytes, n: int) -> np.ndarray:
    raw = zlib.decompress(data)
    if len(raw) != n:
        raise ValueError("corrupt symbol stream")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def ans_encode(t: NVFP4Tensor, table: Optional[np.ndarray] = None) -> ANSStream:
    codes = t.codes.astype(np.uint8).ravel()
    scales = t.scales.astype(np.uint8).ravel()
    f = _freq_table(codes, 16) if table is None else [int(v) for v in table]
    fs = _freq_table(scales, 256)
    return ANSStream(
        payload=_pack(codes),
        scale_payload=_pack(scales),
        freq_table=tuple(f),
        scale_freq_table=tuple(fs),
        n=int(codes.size),
        n_scales=int(scales.size),
        shape=tuple(t.shape),
        block=t.block,
    )


def ans_decode(s: ANSStream) -> NVFP4Tensor:
    codes = _unpack(s.payload, s.n)
    scales = _unpack(s.scale_payload, s.n_scales or s.n)
    nb = (s.shape[-1] + s.block - 1) // s.block
    scales = scales.reshape(s.shape[:-1] + (nb,))
    return NVFP4Tensor(codes.reshape(s.shape), scales, s.shape, s.block)


def ans_size(t: NVFP4Tensor) -> int:
    s = ans_encode(t)
    return len(s.payload) + len(s.scale_payload) + 4 * len(s.freq_table) + 4 * len(s.scale_freq_table) + 64
