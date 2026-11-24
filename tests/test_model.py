import json

from loopkit import ModelPlan, load_plan
from loopkit.cli import main


def test_default_plan_cache_slots_and_applications():
    plan = load_plan()

    assert isinstance(plan, ModelPlan)
    assert plan.n_layers == 24
    assert (plan.loop_start, plan.loop_end, plan.n_passes) == (6, 18, 3)
    assert plan.cache_slots == 48
    assert plan.layer_applications == 48


def test_to_dict_marks_outlier_dim_as_unverified():
    d = load_plan().to_dict()

    assert "outlier_dim" in d
    assert "unverified" in d
    assert "outlier_dim" in d["unverified"]
    assert "unverified" in d["unverified"]["outlier_dim"].lower()


def test_cli_model_exits_zero_and_prints_cache_slots(capsys):
    code = main(["model"])
    plan = json.loads(capsys.readouterr().out)

    assert code == 0
    assert plan["cache_slots"] == 48


def test_custom_config_five_passes(tmp_path, capsys):
    cfg = tmp_path / "plan.toml"
    cfg.write_text(
        '[model]\n'
        'name = "Custom"\n'
        'layers = 24\n'
        'loop_start = 6\n'
        'loop_end = 18\n'
        'loop_passes = 5\n'
        'experts = 128\n'
        'experts_per_token = 8\n'
        'outlier_dim = 2027\n'
        '\n'
        '[codec]\n'
        'nvfp4_layers = 20\n'
        'mxfp8_layers = 4\n'
    )

    plan = load_plan(str(cfg))
    assert plan.cache_slots == 72  # 24 + (5-1)*12

    code = main(["model", "--config", str(cfg)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["cache_slots"] == 72
