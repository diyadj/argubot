import re
import shutil
from pathlib import Path

import pytest
import yaml

from argubot.config import flatten, load_config, merge_configs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "config.yaml"
FOUR_B = ROOT / "configs" / "config_4b.yaml"
TWO_B = ROOT / "configs" / "config_2b.yaml"
OUTPUT_KEYS = ("adapter_dir", "checkpoint_dir", "transcripts_dir", "results_dir", "rating_dir")


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # Scripts always run from the repo root, and `extends:` paths are written that way.
    monkeypatch.chdir(ROOT)


def test_the_main_config_has_no_extends_key():
    cfg = load_config(str(BASE))
    assert "extends" not in cfg
    assert cfg["generator"]["base_model"] == "Qwen/Qwen3.5-0.8B-Base"


def test_the_main_config_keeps_the_settings_the_0_8b_adapter_was_trained_and_run_with():
    cfg = load_config(str(BASE))
    assert cfg["training"]["precision"] == "fp16"
    assert cfg["generator"]["dtype"] == "auto"   # transformers 5.x loads with dtype="auto" when none is given
    assert cfg["paths"]["adapter_dir"] == "outputs/adapter"


@pytest.mark.parametrize("path,model", [(FOUR_B, "Qwen/Qwen3.5-4B-Base"), (TWO_B, "Qwen/Qwen3.5-2B-Base")])
def test_the_bigger_configs_change_only_the_model_precision_and_output_folders(path, model):
    base = dict(flatten(load_config(str(BASE))))
    bigger = dict(flatten(load_config(str(path))))
    assert set(base) == set(bigger)
    changed = {k for k in base if base[k] != bigger[k]}
    assert changed == {
        "paths.adapter_dir",
        "paths.checkpoint_dir",
        "paths.transcripts_dir",
        "paths.results_dir",
        "paths.rating_dir",
        "generator.base_model",
        "generator.dtype",
        "generator.fallback_model",
        "training.precision",
        "training.per_device_batch_size",
        "training.grad_accum_steps",
        "training.eval_batch_size",
    }
    assert bigger["generator.base_model"] == model
    assert bigger["training.precision"] == "bf16"
    assert bigger["generator.fallback_model"] is None


def test_no_two_runs_ever_write_into_the_same_folder():
    runs = {p.name: load_config(str(p))["paths"] for p in (BASE, FOUR_B, TWO_B)}
    folders = [(name, key, Path(paths[key])) for name, paths in runs.items() for key in OUTPUT_KEYS]
    for i, (n1, k1, a) in enumerate(folders):
        for n2, k2, b in folders[i + 1:]:
            assert a != b and a not in b.parents and b not in a.parents, (n1, k1, n2, k2)


def test_a_copied_config_still_loads_from_the_repo_root(tmp_path):
    # scripts/violet_eval.sbatch copies the config into the results folder and edits results_dir.
    copy = tmp_path / "config.yaml"
    shutil.copy(FOUR_B, copy)
    text = copy.read_text()
    lines = [l for l in text.splitlines() if re.match(r"^\s*results_dir:", l)]
    assert len(lines) == 1, "violet_eval.sbatch needs exactly one results_dir line to rewrite"
    copy.write_text(re.sub(r"(?m)^(\s*results_dir:).*$", r"\1 outputs/somewhere_else", text))
    cfg = load_config(str(copy))
    assert cfg["paths"]["results_dir"] == "outputs/somewhere_else"
    assert cfg["generator"]["base_model"] == "Qwen/Qwen3.5-4B-Base"


def test_nested_sections_merge_key_by_key():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merge_configs(base, {"a": {"y": 20}})
    assert out == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}   # the base is not changed in place


def test_a_misspelt_key_is_an_error_not_a_silent_no_op(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"extends": "configs/config.yaml", "training": {"precison": "bf16"}}))
    with pytest.raises(KeyError, match="training.precison"):
        load_config(str(bad))


def test_configs_that_extend_each_other_in_a_loop_are_caught(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nx: 2\n")
    with pytest.raises(ValueError, match="loop"):
        load_config(str(tmp_path / "a.yaml"))


def test_a_missing_parent_says_where_it_looked(tmp_path):
    (tmp_path / "c.yaml").write_text("extends: nowhere.yaml\n")
    with pytest.raises(FileNotFoundError, match="repo root"):
        load_config(str(tmp_path / "c.yaml"))


def test_every_run_trains_with_the_same_effective_batch_of_16():
    for path in (BASE, FOUR_B, TWO_B):
        t = load_config(str(path))["training"]
        assert t["per_device_batch_size"] * t["grad_accum_steps"] == 16, path.name
        assert (t["eval_batch_size"] or t["per_device_batch_size"]) <= t["per_device_batch_size"], path.name
