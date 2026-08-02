from __future__ import annotations

from pathlib import Path

import pytest

from lic_mapping.configuration import load_yaml_config, resolve_config_path


def test_downtown1_configs_load_with_selected_sh_degree() -> None:
    root = Path(__file__).parents[1]
    configs = [load_yaml_config(root / "config" / f"downtown1_sh{degree}.yaml") for degree in range(4)]
    degrees = [config["training"]["sh_degree"] for config in configs]
    assert degrees == [0, 1, 2, 3]
    for config in configs:
        training = config["training"]
        assert training["iterations_per_frame"] == 100
        assert training["keyframe_every"] == 5
        assert training["prune_every_n_keyframes"] == 0
        assert training["scale_multiplier"] == 1.0
        assert training["learning_rate_scales"] == 0.005
        assert training["iteration_decay"] is False


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 1\ntraining:\n  typo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_yaml_config(path)


def test_relative_config_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "run.yaml"
    config_path.parent.mkdir()
    resolved = resolve_config_path("../data/bag", config_path=config_path)
    assert resolved == (tmp_path / "data" / "bag").resolve()
