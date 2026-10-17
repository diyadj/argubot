"""Build the training and validation examples with claim and evidence slots filled (owner: Tricia and Diya).

    python scripts/build_training_data.py

For every bot turn in a train or val chain, replay the chain with the real claim tracker and
the real retriever, exactly as the pipeline does at inference time, then store:
    {"history": [...], "claims": [...], "passages": [...], "response": "..."}
Training on slots that look different from the inference-time slots would hurt systems C1 to D.
Output: cfg["paths"]["train_examples"] and cfg["paths"]["val_examples"].
"""
import argparse

from argubot.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # TODO (Tricia, with Diya's retriever): read cfg["paths"]["cmv_chains"], keep split in
    # {"train", "val"}, replay each chain, and write the two example files with write_jsonl.
    # The test split goes to cfg["paths"]["test_dialogues"] untouched, with user turns only needed.
    raise NotImplementedError("TODO (Tricia): see the docstring")


if __name__ == "__main__":
    main()
