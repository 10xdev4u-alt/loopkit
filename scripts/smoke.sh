#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/test_nvfp4.py::test_ans_roundtrip_is_bit_exact \
  tests/test_bitloop.py::test_certificate_detects_single_bit_flip \
  tests/test_fdm.py::test_fdm_dot_matches_reference
python -m loopkit.cli fdm-check
echo "SMOKE PASS"
