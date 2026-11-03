"""Train the LoRA adapter (owner: Andrea).

    python scripts/train_lora.py --config configs/config_4b.yaml
    python scripts/train_lora.py --config configs/config_4b.yaml --timing-test 30

The timing test trains 30 real steps on the card it lands on (nothing is saved), times a few
validation batches, and projects how long the full run takes and how many 24 h jobs it needs.
It writes the result to outputs/timing/. Run it before the full 4B job.

On the cluster, run it through scripts/violet_train.sbatch instead (docs/UPGRADE_TO_4B.md).

If a run is stopped (time limit, preemption), the same command carries on from the newest
checkpoint. In that case this script exits with code 3, which the batch job uses to resubmit
itself. When training finishes it also measures validation perplexity (reply tokens only,
every slot filled, the same as notebooks/02_train.ipynb) and adds it to training_summary.json.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from argubot.config import load_config
from argubot.generate.training import (
    EXIT_STOPPED_EARLY,
    run_with_smaller_batches,
    train,
    write_training_summary,
)


def _eval_batch(result) -> int:
    return result.eval_batch_size or result.per_device_batch_size


def validation_perplexity(result) -> float:
    from torch.utils.data import DataLoader

    from argubot.eval.metrics import perplexity
    from argubot.generate.training import ReplyOnlyCollator

    collate = ReplyOnlyCollator(result.tokenizer.pad_token_id)

    def measure(bs: int) -> float:
        return perplexity(result.model, DataLoader(result.val_dataset, batch_size=bs, collate_fn=collate))

    ppl, _ = run_with_smaller_batches(measure, _eval_batch(result), label="validation perplexity")
    return ppl


def eval_seconds_per_example(result, n: int = 64) -> float:
    """Forward-pass time per validation example, measured after one untimed warm-up batch."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from argubot.eval.metrics import perplexity
    from argubot.generate.training import ReplyOnlyCollator

    collate = ReplyOnlyCollator(result.tokenizer.pad_token_id)

    def measure(bs: int) -> float:
        count = max(bs, min(n, len(result.val_dataset)))
        perplexity(result.model, DataLoader(Subset(result.val_dataset, range(bs)), batch_size=bs, collate_fn=collate))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        perplexity(result.model, DataLoader(Subset(result.val_dataset, range(count)), batch_size=bs, collate_fn=collate))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.time() - start) / count

    seconds, _ = run_with_smaller_batches(measure, _eval_batch(result), label="the validation timing")
    return seconds


def run_timing_test(cfg, config_path: str, steps: int) -> None:
    import torch

    from argubot.generate.timing import format_report, timing_report

    result = train(cfg, max_steps=steps)
    cuda = torch.cuda.is_available()
    notes = []
    try:
        eval_seconds = eval_seconds_per_example(result)
    except Exception as e:  # noqa: BLE001 - the training numbers are still worth reporting
        # A forward pass costs about a third of a training step, which is close enough to plan with.
        step = sorted(result.step_seconds)[len(result.step_seconds) // 2] if result.step_seconds else 0.0
        eval_seconds = step / (result.per_device_batch_size * result.grad_accum_steps) / 3
        notes.append(f"Could not time validation ({type(e).__name__}), so its hours are an estimate. "
                     "Send the log over before starting the full run.")
    report = timing_report(
        model_name=result.model_name,
        gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
        gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else None,
        precision=result.precision,
        per_device=result.per_device_batch_size,
        accum=result.grad_accum_steps,
        step_seconds=result.step_seconds,
        eval_seconds_per_example=eval_seconds,
        n_train=len(result.train_dataset),
        n_val=len(result.val_dataset),
        configured_epochs=int(cfg["training"]["num_epochs"]),
        train_losses=result.train_losses,
        peak_memory_gb=result.peak_memory_gb,
        fla_installed=importlib.util.find_spec("fla") is not None,
        notes=notes,
    )
    report["eval_batch"] = _eval_batch(result)
    print("\n" + format_report(report))
    out_dir = Path(cfg["paths"]["output_dir"]) / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_slug = "".join(c if c.isalnum() else "-" for c in report["gpu"]).strip("-").lower()
    out = out_dir / f"{Path(config_path).stem}_{gpu_slug}_{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an adapter that is already in paths.adapter_dir")
    ap.add_argument("--skip-perplexity", action="store_true",
                    help="do not measure validation perplexity after training")
    ap.add_argument("--timing-test", type=int, metavar="STEPS", default=None,
                    help="train only this many steps, save nothing, and project the full run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config     : {args.config}")
    print(f"base model : {cfg['generator']['base_model']}")
    print(f"adapter to : {cfg['paths']['adapter_dir']}", flush=True)

    if args.timing_test:
        run_timing_test(cfg, args.config, args.timing_test)
        return

    result = train(cfg, overwrite=args.overwrite)
    if result.stopped_early:
        sys.exit(EXIT_STOPPED_EARLY)

    if not args.skip_perplexity:
        ppl = validation_perplexity(result)
        write_training_summary(result.adapter_dir, {"val_perplexity": round(ppl, 3)})
        print(f"RECORD FOR REPORT -> validation perplexity: {ppl:.2f} (reply tokens, every slot filled)")


if __name__ == "__main__":
    main()
