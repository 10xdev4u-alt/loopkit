import json

import numpy as np

from loopkit.cli import main


def test_health_dense_pair_exits_zero(tmp_path, capsys):
    rng = np.random.default_rng(0)
    n, d_in, d_out = 256, 64, 8
    x = rng.standard_normal((n, d_in))
    np.save(tmp_path / "dense.x.npy", x)
    np.save(tmp_path / "dense.y.npy", x @ rng.standard_normal((d_in, d_out)))

    code = main(["health", "--x", str(tmp_path / "dense.x.npy"),
                 "--y", str(tmp_path / "dense.y.npy"), "--name", "dense"])
    card = json.loads(capsys.readouterr().out)

    assert code == 0
    assert "gamma_hat" in card
    assert card["layer"] == "dense"
    assert card["n_samples"] == n
    assert card["flagged"] is False


def test_health_spiky_pair_exits_one(tmp_path, capsys):
    rng = np.random.default_rng(1)
    n, d_in, d_out = 256, 2304, 32
    x = rng.standard_normal((n, d_in))
    x[:, 2027] *= 20.0
    w = rng.standard_normal((d_in, d_out))
    w[2027] *= 16.0
    np.save(tmp_path / "spiky.x.npy", x)
    np.save(tmp_path / "spiky.y.npy", x @ w)

    code = main(["health", "--x", str(tmp_path / "spiky.x.npy"),
                 "--y", str(tmp_path / "spiky.y.npy"), "--name", "spiky"])
    card = json.loads(capsys.readouterr().out)

    assert code == 1
    assert card["flagged"] is True


def test_health_batch_returns_one_card_per_pair(tmp_path, capsys):
    rng = np.random.default_rng(2)
    for name in ("layer.00", "layer.11"):
        x = rng.standard_normal((64, 12))
        np.save(tmp_path / f"{name}.x.npy", x)
        np.save(tmp_path / f"{name}.y.npy", x @ rng.standard_normal((12, 4)))

    code = main(["health", "--batch", str(tmp_path)])
    cards = json.loads(capsys.readouterr().out)

    assert code == 0
    assert len(cards) == 2
    assert {c["layer"] for c in cards} == {"layer.00", "layer.11"}
