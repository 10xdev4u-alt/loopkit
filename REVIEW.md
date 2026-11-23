# Code review — /tmp/build/loopkit (loopkit)

Scope: `src/loopkit/*.py` (including the new, untracked `shc.py`), `tests/*.py`,
`README.md`, `pyproject.toml`, `kaggle/*.ipynb` and `kaggle_geom*/geometry_gate.ipynb`.

## Methodology

- Read every source file and test in the repo.
- Ran the suite: `PYTHONPATH=src python -m pytest -q -m "not kaggle"` → **22 passed, 1 deselected** (kaggle-marked).
- Targeted probes against the installed package (`PYTHONPATH=/tmp/build/loopkit/src`, numpy 2.5.4, Python 3.14.8):
  - Brute-forced the correct E2M1 mapping (`argmin_c |FP4_LEVELS[c] - ratio|`) and compared it to the code the quantizer stores, for block = [6.0, v] so the scale is 1.0 and the ratio equals v.
  - Padding: `quantize_nvfp4` / `quantize_mxfp8` on shape (2, 20) (20 not divisible by 16); `dequantize_nvfp4` / `dequantize_mxfp8` on a hand-made padded tensor; `ans_encode`/`ans_decode` on the same tensor.
  - Negative zero: `ans_decode(ans_encode(t))` for tensors with code 0 and code 8; `e4m3_encode` brute-forced against nearest valid code over 820 values.
  - `loopedit.apply_edits(diff_passes(a, b)) == b` brute force: 200,000 random float32 trials, 0 failures.
  - `break_even_passes` semantics vs its docstring; `spectral_break_even` band values; `state_digest` vs `certify_pass` on +0.0 vs -0.0.
  - CLI end-to-end: `nvfp4-roundtrip`, `fdm-check`, `certify`, `audit`, `geometry`.
- No files in the repo were modified (tracked files untouched; only the two pre-existing untracked additions `src/loopkit/shc.py` and `tests/test_shc.py` are new to this checkout and were reviewed).

---

## Findings (ranked by severity)

### 1. HIGH — `quantize_nvfp4` maps rounded ratios to the wrong E2M1 codes (src/loopkit/nvfp4.py:108-115)

The codebook is `FP4_LEVELS = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]`
(codes 0-15). The mapping loop

```python
q = np.clip(np.rint(blocks / scales[..., None]).astype(np.int32), -8, 8)
codes = np.zeros(q.shape, dtype=np.uint8)
for i in range(8):
    codes = np.where(q == i, i, codes)
    codes = np.where(q == -i, 8 + i, codes)
```

stores `code = q` for q in 1..7 and `code = 8 + |q|` for q in -7..-1 — i.e. it assumes the level
at code q is q. That is false: the level at code 6 is 4.0, at code 7 is 6.0. Since the scale is
`e4m3(absmax/6)`, the block maximum has ratio ≈ 6 and is stored as **code 6 → decodes to 4.0
instead of 6.0**. Systematic: every block's maximum is under-represented by up to 2 levels;
values with ratios 1-6 decode to about half their true level.

Verified (block = [6.0, v], scale = 1.0, so ratio = v):

| v    | stored code | decoded | correct nearest code (brute force) |
|------|-------------|---------|------------------------------------|
| 6.0  | 6           | 4.0     | 7                                  |
| 4.0  | 4           | 2.0     | 6                                  |
| 3.0  | 3           | 1.5     | 5                                  |
| 2.0  | 2           | 1.0     | 4                                  |
| 1.0  | 1           | 0.5     | 2                                  |
| -6.0 | 14          | -4.0    | 15                                 |
| -4.0 | 12          | -2.0    | 14                                 |
| -3.0 | 11          | -1.5    | 13                                 |

Brute-force correct table for ratio q → code:
`{0:0, 1:2, 2:4, 3:5, 4:6, 5:6, 6:7, 7:7, 8:7, -1:10, -2:12, -3:13, -4:14, -5:14, -6:15, -7:15, -8:15}`
(the loop's table is `{0:8, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, -1:9, -2:10, -3:11, -4:12, -5:13, -6:14, -7:15}`).

Fix — replace the loop with a nearest-code lookup:

```python
ratio = np.rint(blocks / scales[..., None]).astype(np.float64)
codes = np.argmin(np.abs(ratio[..., None] - FP4_LEVELS), axis=-1).astype(np.uint8)
```

(with the padding trim from finding 2). This also changes +0.0 from code 8 to code 0.

### 2. HIGH — `quantize_nvfp4` / `quantize_mxfp8` / `dequantize_nvfp4` / `dequantize_mxfp8` crash on padded shapes

`_block_view` (nvfp4.py:95-99) pads the last axis to a multiple of `block` and returns the pad;
`ans_decode` computes `nb = ceil(shape[-1] / block)` (nvfp4.py:186) — the design intends padded
support. The reshapes in the quantize/dequant paths assume no padding:

- `quantize_nvfp4`: nvfp4.py:115 `codes.reshape(x.shape)` → verified: on shape (2, 20) it raises
  `ValueError: cannot reshape array of size 64 into shape (2,20)`.
- `dequantize_nvfp4`: nvfp4.py:119 `codes.reshape(t.shape[:-1] + (-1, t.block))` →
  `ValueError: cannot reshape array of size 40 into shape (2,newaxis,16)`.
- Same pattern: `quantize_mxfp8` nvfp4.py:134, `dequantize_mxfp8` nvfp4.py:138.

Tests never exercise it: `test_dequant_shape_and_finite` (tests/test_nvfp4.py:33-37) uses
(4, 64, 64) — 64 is a multiple of 16.

Confirming the intent: a hand-made padded tensor (codes (2,20), scales (2,2), shape (2,20),
block 16) roundtrips **bit-exactly** through `ans_encode`/`ans_decode` (codes, scales, shape and
block all equal) — only the quantize/dequant reshapes break.

Fix — store per-element codes (`t.codes.size == prod(t.shape)`) and pad on dequant:

```python
# quantize_nvfp4 (nvfp4.py:115):
codes = codes.reshape(x.shape[:-1] + (-1,))[..., : x.shape[-1]]

# dequantize_nvfp4 (nvfp4.py:118-123):
nb = (t.shape[-1] + t.block - 1) // t.block
codes = t.codes.reshape(t.shape)
padded = np.zeros(t.shape[:-1] + (nb * t.block,), dtype=codes.dtype)
padded[..., : t.shape[-1]] = codes
vals = FP4_LEVELS[padded.reshape(t.shape[:-1] + (nb, t.block)).astype(np.intp)]
out = (vals * e4m3_decode(t.scales)[..., None]).reshape(t.shape[:-1] + (-1,))[..., : t.shape[-1]].reshape(t.shape)

# quantize_mxfp8 / dequantize_mxfp8: same trim/pad pattern at nvfp4.py:134 / 138
```

### 3. MED — ratio 8 / -8 quantizes to code 0 (unmapped, nvfp4.py:108 + loop 112-114)

`np.clip(..., -8, 8)` allows q = 8 and q = -8, but the mapping loop only covers q ∈ {-7..7}
(`np.where(q == -i, 8 + i, codes)` never fires for q = -8, and q = 8 matches nothing), so those
elements keep the default code 0. Verified: `quantize_nvfp4` on a block with absmax 3400
(scale = e4m3(3400/6) saturates at 448; ratio = 3400/448 ≈ 7.6 → q = 8) → codes [0, 8, 8, ...],
dequant [0.0, -0.0, ...] — **the block maximum is lost entirely**.

Fix: fold into finding 1's lookup — with `codes = np.argmin(np.abs(ratio[..., None] - FP4_LEVELS), axis=-1)`
the extremes saturate to the nearest level automatically (verified: nearest code for ratio 8 is 7,
for -8 is 15).

### 4. MED — `break_even_passes` is off by one vs its docstring (geometry.py:33-39)

Docstring: "Smallest n >= 1 where n passes of base work beat codec setup + n*cost."
Implementation: `n = int(np.ceil(codec_setup / (base_per_pass - codec_per_pass))) + 1` (geometry.py:38).

Verified: `break_even_passes(2.0, 1.0, 0.25)` returns 2, but n=1 already satisfies 2.0 > 1.0 + 0.25.
`break_even_passes(2.0, 2.0, 0.5)` returns 3, but n=2 has 4.0 > 2 + 1.0. The smallest n with
`n*(base - codec) > setup` is `floor(setup/(base - codec)) + 1`.
`test_break_even_passes` (tests/test_geometry.py:33-35) codifies the off-by-one.

Fix: `n = int(np.floor(codec_setup / (base_per_pass - codec_per_pass))) + 1`.

### 5. MED — `certify_pass` and `state_digest` disagree on negative zero (bitloop.py:37-75)

`state_digest` hashes the raw bytes (the sign bit of -0.0 is part of the digest), but
`certify_pass` decides "exact" with `np.array_equal`, which treats +0.0 == -0.0 as equal.
Verified: `state_digest({"a": [+0.0]}) != state_digest({"a": [-0.0]})`, while `certify_pass` on those
states reports `exact=True` (`cert.digest == digest(-0.0)`). So `LoopCertificate.all_exact` can be
True while `verify_loop` (bitloop.py:98-102) returns False on the same data. The module's "bitwise"
claim is then not enforced.

Fix: make them agree, e.g. in `certify_pass`:

```python
if not np.array_equal(a, b) or _bytes(a) != _bytes(b):
    ...
```

or compare `state_digest` values instead of `array_equal` for the exact flag.

### 6. MED — `apply_edits` never verifies the edit's digests (loopedit.py:42-49)

`WeightEdit` carries `old_digest`/`new_digest` (computed in `diff_passes`, loopedit.py:37-38), but
`apply_edits` only checks `edit.key in out` and adds `delta`. There is no verification that the edit
is being applied to the state it was diffed from. Reconstruction `out[key] + delta` was bit-exact in
my 200,000-trial brute force, but `a + (b - a) == b` is not guaranteed for floats in general — the
digest check is the real guard and it is not wired to anything.

Fix: in `apply_edits`, before applying:

```python
if hashlib.sha256(np.ascontiguousarray(out[edit.key]).tobytes()).hexdigest() != edit.old_digest:
    raise ValueError("edit applied to a different state than it was diffed from")
```

### 7. LOW — `quantize_nvfp4` stores +0.0 as code 8 (nvfp4.py:109-114)

The loop's `np.where(q == -i, 8 + i, codes)` also fires at i=0 for q == 0, so every zero element is
stored as code 8 (-0.0) even when the input was +0.0 — the comment says "sign of zero preserved as
code 8". Harmless numerically (dequant treats ±0 as equal; the ans roundtrip preserves the stored
code), but it is the wrong code for +0 input.

### 8. LOW — `ans_size` is a heuristic, not an exact byte count (nvfp4.py:191-193)

`ans_size` = `len(payload) + len(scale_payload) + 4*len(freq_table) + 4*len(scale_freq_table) + 64`.
`freq_table` is always length 16 and `scale_freq_table` always length 256 (from `_freq_table`), so the
overhead is a constant 4*16 + 4*256 + 64 = 1152. Verified on a (8, 256) tensor: `ans_size(t) = 2319`
vs actual `len(payload) + len(scale_payload) = 1167`. Not a bug; just don't read it as exact.
(RESULTS.md's "4,194,304 B → 565,073 B" uses the notebook's own payload-only measure.)

### 9. LOW — `fp4_itq` / `geometry_gate` inherit finding 1 (geometry.py:91, 118, 121)

`fp4_itq` and `geometry_gate` call `quantize_nvfp4`; the FP4 MSE numbers and rotation gains in the
geometry gate are computed with the broken quantizer.

### 10. LOW — CLI gaps (cli.py)

- `audit` (cli.py:83-92) reads `name`/`seconds` JSON files, but nothing in the repo produces them
  (`TimingSample` has no JSON writer) — users must hand-craft the files.
- `certify` writes `cert.json` to the CWD (cli.py:69) — fine but surprising.
- Exit codes verified: `nvfp4-roundtrip` → 0/1 on pass/fail; `fdm-check` → 0 (err 0.0);
  `certify` → 1 on non-exact loop; `audit` → 0 when the claim is valid
  (`claim <= ci_low`), 1 otherwise. All consistent with the code.

---

## Notes (not findings)

- `e4m3_encode` (nvfp4.py:47-63): brute-forced vs nearest valid code over 820 values — correct.
  (The one "mismatch" was -0.0 → encoded as 0x80 vs nearest code 0; both decode to zero.)
- `lambda_law` (geometry.py:19-25): correct — recovers c=2.0, lam=0.5 on the test's exact data
  (test passes).
- `spectral_break_even` (geometry.py:42-57): the three asserted bands reproduce the paper values
  (0.3854 / 0.4586 / 0.5061 vs tests' 0.385 / 0.458 / 0.506). Edge semantics: if
  distortion > the maximum attainable tail (~1 − rank/n at g→0, i.e. 0.9375 for n=4096, rank=256),
  the function returns ~0.05 — the lower edge of the search range.
- `joint_itq` (geometry.py:60-72): fine for 2-D passes (its tests); for 1-D passes it returns a
  (1,1) "rotation" rather than (d,d) — undocumented latent behavior (kaggle notebooks always pass 2-D).
- Tests that pass without proving anything: `tests/test_nvfp4.py::test_ans_roundtrip_is_bit_exact`
  and `test_nvfp4_bitwise_roundtrip_1000_vectors` assert only `np.array_equal(t.codes, u.codes)` —
  the roundtrip is trivially true because `ans_encode` is `zlib.compress(codes.tobytes())` and
  `ans_decode` unpacks the same bytes. They never check `dequantize_nvfp4(quantize_nvfp4(x)) ≈ x`.
  There is **no test anywhere** of quantize→dequant fidelity — the missing test would have caught
  finding 1.
- `src/loopkit/shc.py` + `tests/test_shc.py` (new, untracked): reviewed; `recover_weights`
  (dual/primal ridge paths are the correct identities), `_gamma_from_spectrum`, `_lambda_stats`,
  `health_card`, `health_report` — no correctness issues found; its 4 tests pass.
- `fdm` bitwise claim: holds by construction — `fdm_dot` calls `reference_dot` and casts
  (fdm.py:28-36). Verified: `fdm_check` → path "fused-staging", max_abs_err_vs_ref 0.0, exit 0.
- `audit` CI semantics: `verdict = claim_speedup <= ci_low` — conservative claims below the 95%
  lower bound of the ratio distribution are "valid" (README: "A claim is valid only if it is below
  the lower 95% bound, never rounded up"). Verified: `audit --claim 1.2` → exit 0
  (speedup 1.25, ci_low 1.25).
- README "Guarantees": guarantee 1 (ans roundtrip) holds; guarantee 2 (fdm byte-identical) holds.

---

## Checklist answers (nvfp4 quantization, ans_size, bitloop, loopedit, geometry, fdm, audit, CLI)

- nvfp4 quantization mapping (E2M1 levels, negative zero, block scales): **broken** — findings 1, 3, 7.
  Block scales themselves behave (`scale = e4m3(absmax/6)`, floor 2^-9; for absmax=6, scale=1.0, ratio of max=6).
- `ans_size` accounting: heuristic (finding 8).
- `bitloop` digest semantics: consistent between certify/verify except the -0.0 case (finding 5).
- `loopedit` digest verification: absent (finding 6).
- `geometry.lambda_law`: correct. `spectral_break_even`: bands OK, edge semantics note.
  `fp4_itq`/`joint_itq`: inherit finding 1 / note above.
- `fdm` bitwise claim: holds (verified).
- `audit` bootstrap CI: no bug found.
- CLI exit codes: verified consistent.

---

## The README bit-exact claim — the requested paragraph

Is `ans_decode(ans_encode(t)) == t` bit-exact for all shapes including padding and negative zero?

**Yes — the claim as literally stated is true, and I verified it:** `ans_encode` ravels
`t.codes`/`t.scales` into zlib-compressed bytes and `ans_decode` unpacks the same bytes, so codes,
scales, shape and block roundtrip exactly. Verified: the CLI `nvfp4-roundtrip --file demo.npy`
prints `{"lossless": true, ...}` and exits 0 on a random (8, 256) tensor; a hand-made padded tensor
(shape (2,20), block 16, scales (2,2)) roundtrips bit-exactly (codes/scales/shape/block all equal);
code 8 and code 0 both roundtrip unchanged.

Two caveats make the "all shapes including padding" part weaker than it sounds:

1. You cannot reach a padded tensor through the public API: `quantize_nvfp4`/`quantize_mxfp8` crash
   on shapes where `shape[-1] % block != 0` (finding 2), and `dequantize_nvfp4`/`dequantize_mxfp8`
   crash on them too — so "for all shapes including padding" is currently unrealizable via
   `quantize_nvfp4` + `dequantize_nvfp4`; padding only survives in `ans_encode`/`ans_decode` on
   hand-constructed tensors.
2. The claim is about the *stored representation*: it does not imply
   `dequantize(quantize(x)) ≈ x`, and that direction is broken — finding 1:
   `quantize_nvfp4` mis-maps nearly every ratio (verified: value 6.0 → code 6 → decodes to 4.0),
   and `quantize_nvfp4([3400, 0...])` loses the block maximum entirely (finding 3).

---

## Summary

- **CRITICAL: 0**
- **HIGH: 2** — F1 (`quantize_nvfp4` E2M1 mapping is systematically wrong; block max represented at
  67% of its value; value 6.0 → 4.0), F2 (padding crash in quantize/dequant paths).
- **MED: 3** — F3 (ratio 8/-8 → code 0, verified catastrophic loss), F4 (`break_even_passes` off by
  one vs docstring), F5 (`certify_pass`/`state_digest` disagree on -0.0: certificate says exact,
  `verify_loop` would say not).
- **LOW: 4** — F6 (`apply_edits` ignores `old_digest`/`new_digest`), F7 (+0.0 stored as code 8),
  F8 (`ans_size` heuristic), F9 (`fp4_itq`/`geometry_gate` inherit F1).
- **Notes** (not findings): `e4m3_encode` correct; `lambda_law` correct; `spectral_break_even`
  bands match tests (edge semantics); `joint_itq` 1-D behavior; `audit`/`fdm` claims hold; no
  quantize→dequant fidelity test exists; `shc.py` clean; CLI exit codes verified.

**Single worst bug:** F1 — the `quantize_nvfp4` E2M1 code-mapping bug (nvfp4.py:108-115): the
stored code equals the rounded ratio instead of the nearest E2M1 code, so every block's maximum is
under-represented (verified: 6.0 → 4.0; the block max with ratio 6 is stored as level 4.0). It is
silent — no error, and no test anywhere covers quantize→dequant fidelity.
