"""Checkpoints, resuming, the time-limit stop and the out-of-memory retry.

The first group needs nothing but the standard library. The last test is a real (tiny) training
run that is stopped by the same signal Slurm sends before the time limit, then resumed.
"""
import json
import os
import signal

import pytest

from argubot.generate.training import (
    RUN_INFO,
    SUMMARY,
    build_training_kwargs,
    clear_checkpoints,
    fingerprint_differences,
    guard_adapter_dir,
    halve_batch,
    latest_checkpoint,
    prepare_checkpoint_dir,
    record_job_seconds,
    run_fingerprint,
)

CFG = {
    "seed": 42,
    "paths": {"train_examples": "data/processed/train_examples.jsonl"},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "prompt": {"max_history_turns": 6, "max_passage_words": 80},
    "training": {
        "max_seq_len": 1024, "per_device_batch_size": 4, "grad_accum_steps": 4, "learning_rate": 2.0e-4,
        "num_epochs": 3, "warmup_ratio": 0.0, "p_blank_claims": 0.25, "p_blank_evidence": 0.25,
        "max_train_examples": None,
    },
}


def _fp(**training):
    cfg = {**CFG, "training": {**CFG["training"], **training}}
    return run_fingerprint(cfg, "Qwen/Qwen3.5-4B-Base", 18014, 2282, "bf16")


def _checkpoint(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_text("w")
    if complete:
        (d / "trainer_state.json").write_text("{}")
        (d / "optimizer.pt").write_text("o")
    return d


# --- out-of-memory retry --------------------------------------------------------------------

def test_halving_the_batch_keeps_the_effective_batch():
    assert halve_batch(4, 4) == (2, 8)
    assert halve_batch(2, 8) == (1, 16)


def test_an_odd_batch_never_shrinks_the_effective_batch():
    per_device, accum = halve_batch(3, 4)
    assert per_device == 1 and per_device * accum >= 12


def test_a_batch_of_one_cannot_be_halved():
    with pytest.raises(ValueError):
        halve_batch(1, 16)


# --- TrainingArguments names ----------------------------------------------------------------

def test_warmup_ratio_becomes_warmup_steps_on_transformers_5():
    out = build_training_kwargs({"warmup_ratio": 0.03, "seed": 1}, valid={"warmup_steps", "seed"})
    assert out == {"warmup_steps": 0.03, "seed": 1}


def test_zero_warmup_is_simply_left_out():
    assert build_training_kwargs({"warmup_ratio": 0.0}, valid={"warmup_steps"}) == {}


def test_old_transformers_keep_their_own_names():
    out = build_training_kwargs({"warmup_ratio": 0.03, "eval_strategy": "epoch"},
                                valid={"warmup_ratio", "evaluation_strategy"})
    assert out == {"warmup_ratio": 0.03, "evaluation_strategy": "epoch"}


def test_an_unknown_setting_is_an_error_not_a_silent_drop():
    pytest.importorskip("transformers")
    with pytest.raises(TypeError, match="no_such_setting"):
        build_training_kwargs({"no_such_setting": 1}, valid=set())


def test_every_setting_train_uses_is_accepted_by_the_installed_transformers():
    pytest.importorskip("transformers")
    from transformers import TrainingArguments

    from argubot.generate.training import _training_argument_names

    names = ["output_dir", "per_device_train_batch_size", "per_device_eval_batch_size",
             "gradient_accumulation_steps", "learning_rate", "num_train_epochs", "warmup_ratio", "fp16",
             "bf16", "logging_steps", "save_strategy", "save_steps", "save_total_limit", "report_to",
             "remove_unused_columns", "label_names", "eval_strategy", "seed", "max_steps"]
    build_training_kwargs({n: 0.03 if n == "warmup_ratio" else None for n in names},
                          _training_argument_names(TrainingArguments))


# --- which run a checkpoint belongs to ------------------------------------------------------

def test_the_per_device_batch_split_does_not_count_as_a_different_run():
    assert fingerprint_differences(_fp(), _fp(per_device_batch_size=2, grad_accum_steps=8)) == []


def test_a_changed_learning_rate_counts_as_a_different_run():
    diffs = fingerprint_differences(_fp(), _fp(learning_rate=1e-4))
    assert len(diffs) == 1 and "learning_rate" in diffs[0]


# --- the checkpoint folder ------------------------------------------------------------------

def test_a_fresh_start_records_the_run(tmp_path):
    ckpt = tmp_path / "ckpt"
    assert prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True) is None
    info = json.loads((ckpt / RUN_INFO).read_text())
    assert info["fingerprint"] == _fp() and info["train_seconds_so_far"] == 0.0


def test_the_same_run_resumes_from_the_newest_complete_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    _checkpoint(ckpt, 900)
    _checkpoint(ckpt, 1000)
    _checkpoint(ckpt, 1100, complete=False)   # the job died while saving this one
    last = prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    assert last.endswith("checkpoint-1000")   # numeric order, not text order
    assert not (ckpt / "checkpoint-1100").exists()


def test_checkpoints_from_other_settings_are_never_resumed(tmp_path):
    ckpt = tmp_path / "ckpt"
    prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    _checkpoint(ckpt, 100)
    with pytest.raises(RuntimeError, match="learning_rate"):
        prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(learning_rate=1e-4), resume=True)


def test_old_checkpoints_are_not_mixed_into_a_fresh_run(tmp_path):
    ckpt = tmp_path / "ckpt"
    prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    _checkpoint(ckpt, 100)
    with pytest.raises(RuntimeError, match="resume is false"):
        prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=False)


def test_checkpoints_of_unknown_origin_are_refused(tmp_path):
    ckpt = tmp_path / "ckpt"
    _checkpoint(ckpt, 100)
    with pytest.raises(RuntimeError, match=RUN_INFO):
        prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)


def test_checkpoints_cannot_live_inside_the_adapter_folder(tmp_path):
    with pytest.raises(ValueError, match="separate folders"):
        prepare_checkpoint_dir(str(tmp_path / "adapter" / "ckpt"), str(tmp_path / "adapter"), _fp(), resume=True)


def test_training_time_adds_up_over_jobs(tmp_path):
    ckpt = tmp_path / "ckpt"
    prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    record_job_seconds(str(ckpt), 100.0)
    assert record_job_seconds(str(ckpt), 50.0) == 150.0


def test_clearing_removes_only_this_runs_checkpoints(tmp_path):
    ckpt = tmp_path / "ckpt"
    prepare_checkpoint_dir(str(ckpt), str(tmp_path / "adapter"), _fp(), resume=True)
    _checkpoint(ckpt, 100)
    (ckpt / "notes.txt").write_text("keep me")
    clear_checkpoints(str(ckpt))
    assert sorted(p.name for p in ckpt.iterdir()) == ["notes.txt"]
    assert latest_checkpoint(str(ckpt)) is None


def test_a_finished_adapter_is_not_overwritten_by_accident(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    guard_adapter_dir(str(adapter), overwrite=False)          # empty folder is fine
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--overwrite"):
        guard_adapter_dir(str(adapter), overwrite=False)
    guard_adapter_dir(str(adapter), overwrite=True)


# --- a real run, stopped by the time-limit signal and resumed --------------------------------

def test_a_run_stopped_by_the_time_limit_signal_resumes_and_finishes(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("no SIGUSR1 on this platform")
    import copy

    from transformers import TrainerCallback

    from argubot.config import load_config
    from argubot.generate.training import train

    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["generator"]["base_model"] = "sshleifer/tiny-gpt2"
    cfg["paths"].update(
        train_examples="data/toy/toy_train_examples.jsonl",
        val_examples="data/toy/toy_train_examples.jsonl",
        adapter_dir=str(tmp_path / "adapter"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    cfg["training"].update(
        num_epochs=3, per_device_batch_size=2, grad_accum_steps=1, max_seq_len=64,
        precision="fp32", save_steps=1, save_total_limit=2,
    )

    class SlurmSaysTimeIsUp(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                os.kill(os.getpid(), signal.SIGUSR1)

    try:
        first = train(cfg, callbacks=[SlurmSaysTimeIsUp()])
    except OSError as e:  # no network to fetch the tiny model
        pytest.skip(f"model not available offline: {e}")

    assert first.stopped_early
    assert first.checkpoint and os.path.isdir(first.checkpoint)
    assert first.global_step < first.max_steps
    assert not (tmp_path / "adapter" / "adapter_config.json").exists()

    # A second job with a different learning rate must not pick these checkpoints up.
    changed = copy.deepcopy(cfg)
    changed["training"]["learning_rate"] = 1e-3
    with pytest.raises(RuntimeError, match="learning_rate"):
        train(changed)

    second = train(cfg)
    assert not second.stopped_early
    assert second.global_step == second.max_steps == first.max_steps
    saved_at = int(first.checkpoint.rsplit("-", 1)[1])
    assert len(second.step_seconds) == second.max_steps - saved_at   # carried on, did not start over
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    summary = json.loads((tmp_path / "adapter" / SUMMARY).read_text())
    assert summary["jobs"] == 2 and summary["optimizer_steps"] == second.max_steps
    assert not (tmp_path / "ckpt").exists()   # cleared once the adapter is saved

    with pytest.raises(FileExistsError):
        train(cfg)


# --- forward-only passes step down when the GPU is full -------------------------------------

def test_a_forward_pass_that_runs_out_of_memory_is_retried_with_a_smaller_batch():
    torch = pytest.importorskip("torch")
    from argubot.generate.training import run_with_smaller_batches

    tried = []

    def measure(bs):
        tried.append(bs)
        if bs > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (pretend)")
        return 21.6

    assert run_with_smaller_batches(measure, 4, label="a test") == (21.6, 1)
    assert tried == [4, 2, 1]


def test_running_out_of_memory_at_batch_one_is_still_an_error():
    torch = pytest.importorskip("torch")
    from argubot.generate.training import run_with_smaller_batches

    def measure(bs):
        raise torch.OutOfMemoryError("CUDA out of memory (pretend)")

    with pytest.raises(torch.OutOfMemoryError):
        run_with_smaller_batches(measure, 2)
