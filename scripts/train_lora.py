"""Train the LoRA adapter (owner: Andrea).

    python scripts/train_lora.py --config configs/config_4b.yaml

On the cluster, run it through scripts/violet_train.sbatch instead (docs/UPGRADE_TO_4B.md).

If a run is stopped (time limit, preemption), the same command carries on from the newest
checkpoint. In that case this script exits with code 3, which the batch job uses to resubmit
itself. When training finishes it also measures validation perplexity (reply tokens only,
every slot filled, the same as notebooks/02_train.ipynb) and adds it to training_summary.json.
"""
import argparse
import sys

from argubot.config import load_config
from argubot.generate.training import EXIT_STOPPED_EARLY, train, write_training_summary


def validation_perplexity(result) -> float:
    from torch.utils.data import DataLoader

    from argubot.eval.metrics import perplexity
    from argubot.generate.training import ReplyOnlyCollator

    loader = DataLoader(
        result.val_dataset,
        batch_size=result.per_device_batch_size,
        collate_fn=ReplyOnlyCollator(result.tokenizer.pad_token_id),
    )
    return perplexity(result.model, loader)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an adapter that is already in paths.adapter_dir")
    ap.add_argument("--skip-perplexity", action="store_true",
                    help="do not measure validation perplexity after training")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config     : {args.config}")
    print(f"base model : {cfg['generator']['base_model']}")
    print(f"adapter to : {cfg['paths']['adapter_dir']}", flush=True)

    result = train(cfg, overwrite=args.overwrite)
    if result.stopped_early:
        sys.exit(EXIT_STOPPED_EARLY)

    if not args.skip_perplexity:
        ppl = validation_perplexity(result)
        write_training_summary(result.adapter_dir, {"val_perplexity": round(ppl, 3)})
        print(f"RECORD FOR REPORT -> validation perplexity: {ppl:.2f} (reply tokens, every slot filled)")


if __name__ == "__main__":
    main()
