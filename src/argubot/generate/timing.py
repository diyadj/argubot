"""Turn a short timing run into a projection for the full training run.

scripts/train_lora.py --timing-test trains a few dozen real steps on the cluster card, times them,
times a forward pass over some validation examples, and hands the numbers to timing_report().
Everything here is plain arithmetic so it can be tested without a GPU.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

JOB_HOURS = 24.0          # cs425qos maximum per job
USABLE_HOURS = 23.0       # minus model loading, the 20 minute stop signal and a little slack
MAX_RESUBMITS = 3         # scripts/violet_train.sbatch resubmits itself at most this many times


def optimizer_steps(n_train: int, per_device: int, accum: int, epochs: float) -> int:
    """Optimizer steps the Trainer will take: batches per epoch, grouped by accumulation."""
    batches = math.ceil(n_train / per_device)
    return int(math.ceil(batches / accum) * epochs)


def typical_step_seconds(step_seconds: Sequence[float]) -> float:
    """Median step time, leaving out the first two steps (kernel compilation, cache warm-up)."""
    if not step_seconds:
        raise ValueError("No steps were timed.")
    steady = list(step_seconds[2:]) if len(step_seconds) > 4 else list(step_seconds)
    return float(statistics.median(steady))


def project_hours(
    step_seconds: float,
    eval_seconds_per_example: float,
    n_train: int,
    n_val: int,
    per_device: int,
    accum: int,
    epochs: int,
) -> Dict[str, float]:
    steps = optimizer_steps(n_train, per_device, accum, epochs)
    train_h = steps * step_seconds / 3600
    # one validation pass per epoch, plus the perplexity pass train_lora.py runs at the end
    eval_h = eval_seconds_per_example * n_val * (epochs + 1) / 3600
    total_h = train_h + eval_h
    return {
        "epochs": epochs,
        "optimizer_steps": steps,
        "train_hours": round(train_h, 2),
        "eval_hours": round(eval_h, 2),
        "total_hours": round(total_h, 2),
        "jobs": max(1, math.ceil(total_h / USABLE_HOURS)),
    }


def verdict(projection: Dict[str, float], one_job_alternative: Optional[int]) -> str:
    total, jobs = projection["total_hours"], projection["jobs"]
    if jobs == 1 and total <= 20:
        return f"GO: about {total:.1f} h, fits in one 24 h job."
    if jobs == 1:
        return (f"GO, but close to the limit: about {total:.1f} h. If it runs over, the job saves a "
                "checkpoint and resubmits itself.")
    if jobs <= MAX_RESUBMITS + 1:
        msg = (f"GO with resubmits: about {total:.1f} h, so {jobs} jobs back to back. The batch job "
               "saves and resubmits itself, but nobody else can use the GPU meanwhile.")
        if one_job_alternative:
            msg += f" Or train {one_job_alternative} epoch(s) instead, which fits in one job."
        return msg
    return (f"NO GO: about {total:.1f} h ({jobs} jobs). Check that flash-linear-attention is installed, "
            "ask for a faster card, or use Qwen/Qwen3.5-2B-Base instead.")


def timing_report(
    *,
    model_name: str,
    gpu: str,
    gpu_memory_gb: Optional[float],
    precision: str,
    per_device: int,
    accum: int,
    step_seconds: Sequence[float],
    eval_seconds_per_example: float,
    n_train: int,
    n_val: int,
    configured_epochs: int,
    train_losses: Sequence[float],
    peak_memory_gb: Optional[float],
    fla_installed: bool,
    notes: Sequence[str] = (),
) -> Dict[str, Any]:
    step = typical_step_seconds(step_seconds)
    projections = [
        project_hours(step, eval_seconds_per_example, n_train, n_val, per_device, accum, e)
        for e in sorted({1, 2, 3, int(configured_epochs)})
    ]
    configured = next(p for p in projections if p["epochs"] == int(configured_epochs))
    fits = [p["epochs"] for p in projections if p["jobs"] == 1 and p["epochs"] < configured_epochs]
    warnings: List[str] = list(notes)
    if not train_losses:
        warnings.append("No training loss was logged, so there is no check that it stays finite. Run the "
                        "timing test again with more steps before starting the full run.")
    elif not all(math.isfinite(x) for x in train_losses):
        warnings.append("The training loss is not finite (NaN or inf). Do not start the full run: check "
                        "training.precision (bf16 on a cluster card) before anything else.")
    if peak_memory_gb and gpu_memory_gb and peak_memory_gb > 0.92 * gpu_memory_gb:
        warnings.append(f"Peak memory {peak_memory_gb:.1f} GB is close to the card's {gpu_memory_gb:.0f} GB. "
                        "The out-of-memory retry will handle it, but expect a smaller batch.")
    if not fla_installed and "qwen3.5" in model_name.lower():
        warnings.append("flash-linear-attention is not installed, so these times are about 10x slower than "
                        "they need to be. Install it and run the timing test again.")
    return {
        "model": model_name,
        "gpu": gpu,
        "gpu_memory_gb": round(gpu_memory_gb, 1) if gpu_memory_gb else None,
        "precision": precision,
        "batch": f"{per_device} x {accum}",
        "seconds_per_optimizer_step": round(step, 3),
        "steps_timed": len(step_seconds),
        "eval_seconds_per_example": round(eval_seconds_per_example, 4),
        "peak_memory_gb": round(peak_memory_gb, 2) if peak_memory_gb else None,
        "train_losses": [round(x, 4) for x in train_losses],
        "flash_linear_attention": fla_installed,
        "projections": projections,
        "verdict": verdict(configured, max(fits) if fits else None),
        "warnings": warnings,
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        "=== timing test ===",
        f"model            : {report['model']}",
        f"gpu              : {report['gpu']} ({report['gpu_memory_gb']} GB)",
        f"precision, batch : {report['precision']}, {report['batch']}",
        f"step time        : {report['seconds_per_optimizer_step']} s per optimizer step ({report['steps_timed']} timed)",
        f"peak memory      : {report['peak_memory_gb']} GB",
        f"train loss       : {report['train_losses']}",
        f"flash-linear-attn: {'installed' if report['flash_linear_attention'] else 'MISSING'}",
        "",
        "epochs  steps   train h  eval h  total h  jobs",
    ]
    for p in report["projections"]:
        lines.append(f"{p['epochs']:>6}  {p['optimizer_steps']:>5}  {p['train_hours']:>8}  {p['eval_hours']:>6}  "
                     f"{p['total_hours']:>7}  {p['jobs']:>4}")
    lines += ["", f"VERDICT: {report['verdict']}"]
    lines += [f"WARNING: {w}" for w in report["warnings"]]
    return "\n".join(lines)
