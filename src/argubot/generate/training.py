"""Training pieces for the LoRA adapter (owner: Andrea).

Written and tested here: reply-only labels, batch padding, the dataset that blanks slots
at random every epoch, and the LoRA config. Left for Andrea: the actual Trainer run in train().
"""
from __future__ import annotations

import random
from dataclasses import dataclass
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


@dataclass
class TrainResult:
    """What train() hands back so a notebook can evaluate without rebuilding anything."""

    model: Any
    tokenizer: Any
    train_dataset: Any
    val_dataset: Any
    trainable_params: int
    total_params: int
    train_seconds: float
    model_name: str
    adapter_dir: str
    device: str


def _load_base(model_name: str, dtype):
    """Load tokenizer + causal LM, tolerant of the torch_dtype -> dtype rename and a missing pad token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:            # only set pad if the tokenizer has none
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"         # keep the [response] tag; drop the oldest text instead
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:                    # older transformers still use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return tok, model


def _training_argument_names(training_arguments_cls) -> set:
    """The kwargs a given transformers TrainingArguments accepts; the field set varies by version."""
    import dataclasses
    import inspect

    names: set = set()
    try:
        names |= {f.name for f in dataclasses.fields(training_arguments_cls)}
    except TypeError:
        pass
    try:
        names |= set(inspect.signature(training_arguments_cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):
        pass
    return names


def train(cfg: Dict[str, Any]) -> "TrainResult":
    """Fine-tune the LoRA adapter, save it, and return the model and datasets for evaluation.

    Follows the recipe the pieces above were built for: reply-only loss, slots blanked at random
    each epoch on train and left filled on val, evaluation each epoch, and the adapter saved alone.
    Under fp16 the LoRA params are kept in fp32 to avoid the PEFT "unscale FP16 gradients" error.
    """
    import time

    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    from argubot.io import read_jsonl

    set_seed(cfg["seed"])
    t = cfg["training"]
    paths = cfg["paths"]
    use_fp16 = bool(t["fp16"]) and torch.cuda.is_available()   # a T4 has no bf16
    dtype = torch.float16 if use_fp16 else torch.float32

    # 1. Base model + tokenizer, falling back if the configured model will not load.
    model_name = cfg["generator"]["base_model"]
    try:
        tokenizer, model = _load_base(model_name, dtype)
    except Exception as e:  # noqa: BLE001 - report and fall back rather than crash the run
        print(f"Could not load {model_name!r}: {type(e).__name__}: {e}")
        model_name = cfg["generator"]["fallback_model"]
        print(f"Falling back to {model_name!r} -- tell the group.")
        tokenizer, model = _load_base(model_name, dtype)

    # GPT-2 names its attention projection c_attn; all-linear is for Qwen3.5.
    if getattr(model.config, "model_type", "") == "gpt2":
        cfg = {**cfg, "lora": {**cfg["lora"], "target_modules": ["c_attn"]}}
        print("GPT-2 detected: using target_modules=['c_attn'].")

    # 2. LoRA. Keep trainable params in fp32 so fp16 training does not hit the unscale error.
    model = get_peft_model(model, build_lora_config(cfg))
    if use_fp16:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Memory: gradient checkpointing trades a little compute for a large drop in activation
    # memory, which a 24 GB card needs at seq_len 1024 (especially a shared one). use_cache is
    # incompatible with it, and a LoRA (frozen base) needs inputs to require grad for it to work.
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3. Datasets. Train blanks slots at random each epoch; val keeps every slot filled.
    train_examples = list(read_jsonl(paths["train_examples"]))
    if t.get("max_train_examples"):
        train_examples = train_examples[: t["max_train_examples"]]
    val_examples = list(read_jsonl(paths["val_examples"]))
    train_ds = ReplyOnlyDataset(train_examples, tokenizer, cfg, train=True, seed=cfg["seed"])
    val_ds = ReplyOnlyDataset(val_examples, tokenizer, cfg, train=False, seed=cfg["seed"])

    # 4. Trainer: eval each epoch, gradient accumulation and warmup from the config.
    # Keep only the kwargs this transformers version actually accepts. Field names drift between
    # releases (e.g. evaluation_strategy -> eval_strategy, and some are dropped entirely), so pass
    # what is supported and warn about the rest instead of crashing.
    desired = dict(
        output_dir=paths["adapter_dir"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=t["num_epochs"],
        warmup_ratio=t["warmup_ratio"],
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="no",           # the adapter is saved by hand at the end, not as checkpoints
        report_to="none",
        remove_unused_columns=False,  # our dataset yields dicts the collator consumes as-is
        label_names=["labels"],
        eval_strategy="epoch",        # evaluate once per epoch
        seed=cfg["seed"],
    )
    valid = _training_argument_names(TrainingArguments)
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        desired["evaluation_strategy"] = desired.pop("eval_strategy")  # pre-4.46 name
    accepted = {k: v for k, v in desired.items() if k in valid}
    dropped = sorted(set(desired) - set(accepted))
    if dropped:
        import transformers

        print(f"Note: transformers {transformers.__version__} does not accept {dropped}; dropped them.")
    args = TrainingArguments(**accepted)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=ReplyOnlyCollator(tokenizer.pad_token_id),
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    model.config.use_cache = True  # re-enable for fast generation after training

    # 5. Save the adapter only (PEFT writes adapter_config.json + the LoRA weights).
    adapter_dir = paths["adapter_dir"]
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    device = str(getattr(model, "device", "cpu"))
    print(f"\nRECORD FOR REPORT -> trainable: {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")
    print(f"RECORD FOR REPORT -> train time: {train_seconds:.1f}s on {device}")
    print(f"adapter saved to {adapter_dir}")

    return TrainResult(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        trainable_params=trainable,
        total_params=total,
        train_seconds=train_seconds,
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
    )
