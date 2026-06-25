from pathlib import Path

from colorfm.utils.experiment import apply_overrides, load_yaml


def test_apply_dotted_overrides():
    cfg = {"model": {"d_model": 32}, "training": {"epochs": 1}}
    out = apply_overrides(cfg, {"model.d_model": 64, "training.epochs": 2})
    assert out["model"]["d_model"] == 64
    assert out["training"]["epochs"] == 2
    assert cfg["model"]["d_model"] == 32


def test_sweep_manifest_exists():
    sweep = load_yaml(Path("configs/sweep_quick.yaml"))
    assert "experiments" in sweep
    assert len(sweep["experiments"]) >= 3
