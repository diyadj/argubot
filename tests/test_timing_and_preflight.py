import json
from pathlib import Path

import pytest

from argubot.generate.timing import (
    format_report,
    optimizer_steps,
    project_hours,
    timing_report,
    typical_step_seconds,
    verdict,
)
from argubot.preflight import check_stage, failed

ROOT = Path(__file__).resolve().parents[1]


# --- timing projection ----------------------------------------------------------------------

def test_the_full_run_is_3378_optimizer_steps():
    # 18,014 examples, batch 4 x 4, 3 epochs: what the 0.8B run did and the 4B run will do
    assert optimizer_steps(18014, 4, 4, 3) == 3378
    assert optimizer_steps(18014, 2, 8, 3) == 3378


def test_the_first_two_steps_do_not_count():
    assert typical_step_seconds([30.0, 20.0, 10.0, 10.0, 11.0, 9.0]) == 10.0


def test_projection_adds_validation_and_counts_jobs():
    p = project_hours(10.0, 0.1, 18014, 2282, 4, 4, 3)
    assert p["optimizer_steps"] == 3378
    assert p["train_hours"] == pytest.approx(9.38, abs=0.01)
    assert p["eval_hours"] == pytest.approx(0.25, abs=0.01)
    assert p["jobs"] == 1


def test_verdicts():
    assert verdict({"total_hours": 9.6, "jobs": 1}, None).startswith("GO:")
    assert "close to the limit" in verdict({"total_hours": 22.0, "jobs": 1}, None)
    msg = verdict({"total_hours": 40.0, "jobs": 2}, 2)
    assert "2 jobs" in msg and "2 epoch(s)" in msg
    assert verdict({"total_hours": 120.0, "jobs": 6}, None).startswith("NO GO")


def _report(**overrides):
    kw = dict(model_name="Qwen/Qwen3.5-4B-Base", gpu="NVIDIA L40S", gpu_memory_gb=44.5, precision="bf16",
              per_device=4, accum=4, step_seconds=[20, 15] + [10.0] * 10, eval_seconds_per_example=0.1,
              n_train=18014, n_val=2282, configured_epochs=3, train_losses=[3.1, 2.9],
              peak_memory_gb=20.0, fla_installed=True)
    kw.update(overrides)
    return timing_report(**kw)


def test_a_healthy_report_has_no_warnings():
    r = _report()
    assert r["warnings"] == [] and r["verdict"].startswith("GO")
    assert "VERDICT" in format_report(r)
    json.dumps(r)   # it is saved as JSON


def test_nan_loss_missing_kernel_and_tight_memory_are_flagged():
    r = _report(train_losses=[float("nan")], fla_installed=False, peak_memory_gb=44.0)
    text = " ".join(r["warnings"])
    assert "not finite" in text and "flash-linear-attention" in text and "close to the card" in text


# --- preflight ------------------------------------------------------------------------------

GPU_OK = {"available": True, "name": "NVIDIA L40S", "memory_gb": 44.5, "bf16": True}


def _cfg(tmp_path, **paths):
    """The 4B config with every path moved under tmp_path, so nothing real is read or written."""
    import os

    from argubot.config import load_config

    old = os.getcwd()
    os.chdir(ROOT)   # extends: paths are written relative to the repo root
    try:
        cfg = load_config("configs/config_4b.yaml")
    finally:
        os.chdir(old)
    base = {k: str(tmp_path / Path(v)) for k, v in cfg["paths"].items()}
    base.update(paths)
    cfg["paths"] = base
    return cfg


def _statuses(checks, what):
    return [c.status for c in checks if c.what == what]


def test_train_stage_fails_without_training_data(tmp_path):
    checks = check_stage(_cfg(tmp_path), "train", gpu=GPU_OK)
    assert failed(checks) and "FAIL" in _statuses(checks, "train_examples")


def test_train_stage_passes_with_data_and_a_free_adapter_folder(tmp_path):
    cfg = _cfg(tmp_path, train_examples=str(ROOT / "data/toy/toy_train_examples.jsonl"),
               val_examples=str(ROOT / "data/toy/toy_train_examples.jsonl"))
    checks = check_stage(cfg, "train", gpu=GPU_OK)
    assert not failed(checks), [c.line() for c in checks]


def test_train_stage_protects_a_finished_adapter(tmp_path):
    cfg = _cfg(tmp_path, train_examples=str(ROOT / "data/toy/toy_train_examples.jsonl"),
               val_examples=str(ROOT / "data/toy/toy_train_examples.jsonl"))
    Path(cfg["paths"]["adapter_dir"]).mkdir(parents=True)
    (Path(cfg["paths"]["adapter_dir"]) / "adapter_config.json").write_text("{}")
    assert failed(check_stage(cfg, "train", gpu=GPU_OK))
    assert not failed(check_stage(cfg, "train", gpu=GPU_OK, overwrite=True))


def test_train_stage_spots_checkpoints_from_other_settings(tmp_path):
    from argubot.generate.training import prepare_checkpoint_dir, run_fingerprint

    toy = str(ROOT / "data/toy/toy_train_examples.jsonl")
    cfg = _cfg(tmp_path, train_examples=toy, val_examples=toy)
    fp = run_fingerprint(cfg, cfg["generator"]["base_model"], 6, 6, "bf16")
    fp["training"]["learning_rate"] = 1e-3
    ckpt = Path(cfg["paths"]["checkpoint_dir"])
    prepare_checkpoint_dir(str(ckpt), cfg["paths"]["adapter_dir"], fp, resume=True)
    d = ckpt / "checkpoint-100"
    d.mkdir()
    for name in ("adapter_model.safetensors", "trainer_state.json", "optimizer.pt"):
        (d / name).write_text("x")
    checks = check_stage(cfg, "train", gpu=GPU_OK)
    assert "FAIL" in _statuses(checks, "checkpoints")


def test_bf16_on_a_card_without_it_fails_and_no_gpu_is_only_skipped(tmp_path):
    toy = str(ROOT / "data/toy/toy_train_examples.jsonl")
    cfg = _cfg(tmp_path, train_examples=toy, val_examples=toy)
    t4 = {"available": True, "name": "Tesla T4", "memory_gb": 15.0, "bf16": False}
    assert "FAIL" in _statuses(check_stage(cfg, "timing", gpu=t4), "bf16")
    login = check_stage(cfg, "timing", gpu={"available": False})
    assert _statuses(login, "GPU") == ["SKIP"] and not failed(login)


def test_run_stage_refuses_the_0_8b_adapter_with_the_4b_config(tmp_path):
    cfg = _cfg(tmp_path)
    adapter = Path(cfg["paths"]["adapter_dir"])
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base"}))
    checks = check_stage(cfg, "run", gpu=GPU_OK)
    assert "FAIL" in _statuses(checks, "adapter")
    assert "FAIL" in _statuses(checks, "passages")
    assert "FAIL" in _statuses(checks, "bm25 index")


def test_run_stage_reports_progress_and_catches_partial_transcripts(tmp_path):
    cfg = _cfg(tmp_path, test_dialogues=str(ROOT / "data/toy/toy_dialogues.jsonl"))
    n = sum(1 for _ in open(cfg["paths"]["test_dialogues"]))
    tdir = Path(cfg["paths"]["transcripts_dir"])
    tdir.mkdir(parents=True)
    (tdir / "A.jsonl").write_text("{}\n" * n)
    (tdir / "B.jsonl").write_text("{}\n" * (n - 1 if n > 1 else 2))
    checks = check_stage(cfg, "run", gpu=GPU_OK)
    lines = " ".join(c.line() for c in checks if c.what == "transcripts")
    assert "done ['A']" in lines and "FAIL" in _statuses(checks, "transcripts")


def test_eval_stage_needs_all_five_systems(tmp_path):
    cfg = _cfg(tmp_path, test_dialogues=str(ROOT / "data/toy/toy_dialogues.jsonl"))
    n = sum(1 for _ in open(cfg["paths"]["test_dialogues"]))
    tdir = Path(cfg["paths"]["transcripts_dir"])
    tdir.mkdir(parents=True)
    for s in ("A", "B", "C1", "C2"):
        (tdir / f"{s}.jsonl").write_text("{}\n" * n)
    assert failed(check_stage(cfg, "eval"))
    (tdir / "D.jsonl").write_text("{}\n" * n)
    assert not failed(check_stage(cfg, "eval"))
