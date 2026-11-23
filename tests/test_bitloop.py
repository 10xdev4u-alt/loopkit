import numpy as np
import pytest

from loopkit.bitloop import assert_lossless, certify_loop, state_digest, verify_loop
from loopkit.loopedit import apply_edits, diff_passes


def _state(seed, key="attn.q"):
    return {key: np.random.default_rng(seed).standard_normal((8, 16)).astype(np.float32)}


def test_certificate_detects_single_bit_flip():
    refs = [_state(100 + i) for i in range(3)]
    cands = [{k: v.copy() for k, v in s.items()} for s in refs]
    cert = certify_loop(refs, cands)
    assert cert.all_exact
    assert_lossless(cert)

    raw = cands[1]["attn.q"].view(np.uint8)
    raw.flat[0] ^= 1
    bad = certify_loop(refs, cands)
    assert not bad.all_exact
    assert any(not p.exact and p.n_mismatch > 0 for p in bad.passes)
    with pytest.raises(AssertionError, match=bad.root_digest):
        assert_lossless(bad)
    assert not verify_loop(cert, cands)


def test_state_digest_is_order_independent():
    a = _state(1)["attn.q"]
    b = _state(2)["attn.q"]
    assert state_digest({"a": a, "b": b}) == state_digest({"b": b, "a": a})


def test_loopedit_is_immutable():
    a = _state(5); b = {k: v + 1 for k, v in a.items()}
    edits = diff_passes(a, b, pass_index=1)
    assert edits and all(e.pass_index == 1 for e in edits)
    out = apply_edits(a, edits)
    assert np.array_equal(out["attn.q"], b["attn.q"])
    assert not np.array_equal(a["attn.q"], b["attn.q"])
