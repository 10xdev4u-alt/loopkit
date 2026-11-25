import json

import pytest

from loopkit import TrafficReport, sweep_report, union_report
from loopkit.cli import main


def test_union_report_default_load_counts():
    report = union_report(8, msg_scale=0.1)

    assert isinstance(report, TrafficReport)
    assert (report.loads_off, report.loads_on) == (24, 8)
    assert report.reduction == pytest.approx(0.667)
    assert report.msg_scale == pytest.approx(0.1)


def test_union_report_rejects_out_of_range_union():
    with pytest.raises(ValueError):
        union_report(4)
    with pytest.raises(ValueError):
        union_report(25)


def test_cli_traffic_exits_zero_and_prints_reduction(capsys):
    code = main(["traffic", "--union", "8", "--msg-scale", "0.1"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["reduction"] == pytest.approx(0.667)
    assert payload["loads_off"] == 24


def test_sweep_reduction_decreases_as_scale_grows():
    sweep = sweep_report({"0.05": 8.25, "0.1": 8.48, "0.3": 9.29})

    assert len(sweep) == 3
    reductions = [row["reduction"] for row in sweep.values()]
    assert reductions == sorted(reductions, reverse=True)
    assert len(set(reductions)) == 3
    assert sweep["0.1"]["msg_scale"] == pytest.approx(0.1)


def test_cli_traffic_sweep_file_exits_zero(tmp_path, capsys):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"0.05": 8.25, "0.1": 8.48, "0.3": 9.29}))

    code = main(["traffic", "--sweep-file", str(path)])
    sweep = json.loads(capsys.readouterr().out)

    assert code == 0
    assert len(sweep) == 3
