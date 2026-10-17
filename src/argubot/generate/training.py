"""Training pieces for the LoRA adapter (owner: Andrea).

Written and tested here: reply-only labels, batch padding, the dataset that blanks slots
at random every epoch, and the LoRA config. Left for Andrea: the actual Trainer run in train().
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from argubot.prompt import training_example_to_pair

IGNORE_INDEX = -100


def encode_example(tokenizer, prompt: str, response: str, max_len: int) -> Dict[str, List[int]]:
    """Tokenize prompt and reply separately so the loss only covers the reply.

    The prompt ends in a newline and the reply starts right after it, which matches
    how generation continues from a tokenized prompt at inference time. The EOS token
    stays in the labels so the model learns to stop.

    Too long? The oldest prompt tokens are dropped first. That can clip the [claims] slot,
    so keep max_seq_len generous or shorten history and passages in the prompt config instead.
    """
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    response_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)

    if len(response_ids) >= max_len:
        response_ids = response_ids[:max_len]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(max_len - len(response_ids)) :]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def pad_batch(features: Sequence[Dict[str, List[int]]], pad_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad to the longest item in the batch. Padded labels are ignored by the loss."""
    longest = max(len(f["input_ids"]) for f in features)
    batch: Dict[str, List[List[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        gap = longest - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_id] * gap)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * gap)
        batch["attention_mask"].append(f["attention_mask"] + [0] * gap)
    return batch


class ReplyOnlyCollator:
    """Use this instead of DataCollatorForLanguageModeling.

    That collator rebuilds labels from input_ids and masks every pad token. When pad equals
    EOS (the GPT-2 habit in the base notebook) it also masks the real EOS, so the model never
    learns to end a reply. It also trains on the prompt, which the proposal says we do not do.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        batch = pad_batch(features, self.pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class ReplyOnlyDataset:
    """Encodes lazily so slot blanking is redrawn every epoch instead of frozen once.

    Pass train=False for validation so every slot stays filled.
    """

    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        tokenizer,
        cfg: Dict[str, Any],
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        t = self.cfg["training"]
        prompt, response = training_example_to_pair(
            self.examples[i],
            self.cfg["prompt"],
            rng=self.rng if self.train else None,
            p_blank_claims=t["p_blank_claims"],
            p_blank_evidence=t["p_blank_evidence"],
        )
        return encode_example(self.tokenizer, prompt, response, t["max_seq_len"])


def build_lora_config(cfg: Dict[str, Any]):
    from peft import LoraConfig, TaskType

    lora = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],  # "all-linear" for Qwen3.5, ["c_attn"] for GPT-2
    )


def train(cfg: Dict[str, Any]) -> None:
    """TODO (Andrea): wire the pieces above into a Trainer run.

      1. Load the config model and tokenizer, set pad_token only if it is missing.
      2. get_peft_model(model, build_lora_config(cfg)), then print_trainable_parameters()
         and write that number down for the report (about 1% of the weights).
      3. Build ReplyOnlyDataset for train (train=True) and validation (train=False).
      4. Trainer with ReplyOnlyCollator, eval each epoch, and log the settings you tuned.
      5. model.save_pretrained(cfg["paths"]["adapter_dir"]) saves the adapter only.
      6. Record training time and GPU so the report can state them.
    """
    raise NotImplementedError("TODO (Andrea): see the steps in this docstring")
