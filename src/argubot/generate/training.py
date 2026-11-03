"""Training pieces for the LoRA adapter (owner: Andrea).

Reply-only labels, batch padding, the dataset that blanks slots at random every epoch, the LoRA
config, and train() itself, which also handles precision (fp16, bf16 or fp32), checkpoints and
resuming, a clean stop before the cluster's time limit, and an out-of-memory retry.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


PRECISIONS = ("bf16", "fp16", "fp32")


def resolve_precision(training_cfg: Dict[str, Any], cuda_available: bool, bf16_supported: bool) -> str:
    """Pick 'bf16', 'fp16' or 'fp32' for this run from training.precision.

    Older configs had `fp16: true/false` instead, so that is still understood. Without a GPU
    everything runs in fp32 (CPU tests). Asking for bf16 on a card that cannot do it (a T4 or a
    V100) is an error rather than a quiet switch to fp16, because fp16 is exactly what we are
    trying to avoid on the bigger model, and finding out after ten hours of NaNs is worse.
    """
    wanted = training_cfg.get("precision")
    if wanted is None:
        wanted = "fp16" if training_cfg.get("fp16") else "fp32"
    wanted = str(wanted).lower()
    if wanted not in PRECISIONS:
        raise ValueError(f"training.precision must be one of {PRECISIONS}, got {wanted!r}")
    if not cuda_available:
        return "fp32"
    if wanted == "bf16" and not bf16_supported:
        raise RuntimeError(
            "training.precision is bf16 but this GPU cannot do bf16 (a T4 or V100). Ask for an RTX 3090, "
            "A40, L40S or H100 (sbatch --constraint), or set training.precision: fp16 in the config."
        )
    return wanted


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


# ---------------------------------------------------------------------------------------------
# TrainingArguments. transformers renames and drops fields between releases. A dropped field used
# to be skipped with a one-line note, which is how warmup_ratio went missing on transformers 5.x
# (it became warmup_steps). Known renames are mapped now, and anything else unsupported is an
# error, so a setting can no longer vanish without anyone noticing.

def build_training_kwargs(desired: Dict[str, Any], valid: set) -> Dict[str, Any]:
    out = dict(desired)
    if "eval_strategy" in out and "eval_strategy" not in valid and "evaluation_strategy" in valid:
        out["evaluation_strategy"] = out.pop("eval_strategy")          # before transformers 4.46
    if "warmup_ratio" in out and "warmup_ratio" not in valid and "warmup_steps" in valid:
        ratio = float(out.pop("warmup_ratio"))                          # transformers 5.x: a float
        if ratio > 0:                                                   # below 1 is read as a ratio
            out["warmup_steps"] = ratio
    unsupported = sorted(set(out) - valid)
    if unsupported:
        import transformers

        raise TypeError(
            f"transformers {transformers.__version__} does not accept {unsupported} in TrainingArguments. "
            "Map the new name in build_training_kwargs() rather than dropping the setting."
        )
    return out


# ---------------------------------------------------------------------------------------------
# Checkpoints and resuming. The helpers are plain functions so they can be tested without a GPU.

EXIT_STOPPED_EARLY = 3            # train_lora.py exits with this when a job should be resubmitted
RUN_INFO = "run_info.json"        # in the checkpoint folder: which run the checkpoints belong to
SUMMARY = "training_summary.json"  # in the adapter folder: numbers for the report
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def halve_batch(per_device: int, accum: int) -> Tuple[int, int]:
    """Half the batch per device, more accumulation, same effective batch. Used after running out of memory."""
    if per_device <= 1:
        raise ValueError("The batch is already 1 per device, so there is nothing left to halve.")
    effective = per_device * accum
    smaller = per_device // 2
    return smaller, -(-effective // smaller)   # round accumulation up so the effective batch never shrinks


def run_fingerprint(
    cfg: Dict[str, Any], model_name: str, n_train: int, n_val: int, precision: str
) -> Dict[str, Any]:
    """Everything that has to match before a checkpoint can be resumed.

    The per-device batch and grad_accum are left out on purpose: the out-of-memory fallback swaps
    one for the other, and only their product changes the training.
    """
    t = cfg["training"]
    keys = ("max_seq_len", "learning_rate", "num_epochs", "warmup_ratio",
            "p_blank_claims", "p_blank_evidence", "max_train_examples")
    return json.loads(json.dumps({
        "base_model": model_name,
        "precision": precision,
        "seed": cfg["seed"],
        "lora": cfg["lora"],
        "prompt": cfg["prompt"],
        "train_examples": cfg["paths"]["train_examples"],
        "n_train": n_train,
        "n_val": n_val,
        "effective_batch": t["per_device_batch_size"] * t["grad_accum_steps"],
        "training": {k: t.get(k) for k in keys},
    }))


def fingerprint_differences(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    from argubot.config import flatten

    a, b = dict(flatten(old)), dict(flatten(new))
    return [f"{k}: {a.get(k)!r} in the checkpoints, {b.get(k)!r} now"
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def is_complete_checkpoint(path: Path) -> bool:
    """A job killed in the middle of saving leaves a folder without these, which cannot be resumed."""
    weights = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")
    return (any((path / w).is_file() for w in weights)
            and (path / "trainer_state.json").is_file()
            and (path / "optimizer.pt").is_file())


def _checkpoint_folders(checkpoint_dir: str) -> List[Tuple[int, Path]]:
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in d.iterdir() if p.is_dir() and (m := _CHECKPOINT_RE.match(p.name))]
    return sorted(found)


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """The newest checkpoint that was written completely, or None."""
    complete = [p for _, p in _checkpoint_folders(checkpoint_dir) if is_complete_checkpoint(p)]
    return str(complete[-1]) if complete else None


def _read_run_info(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    path = Path(checkpoint_dir) / RUN_INFO
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_run_info(checkpoint_dir: str, info: Dict[str, Any]) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tmp = Path(checkpoint_dir) / (RUN_INFO + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, Path(checkpoint_dir) / RUN_INFO)   # never leave a half-written file behind


def prepare_checkpoint_dir(
    checkpoint_dir: str, adapter_dir: str, fingerprint: Dict[str, Any], resume: bool
) -> Optional[str]:
    """Check the checkpoint folder before training and return the checkpoint to resume from, if any.

    Refuses to mix runs: checkpoints made with different settings are never resumed, and old
    checkpoints are never silently mixed into a fresh run.
    """
    ckpt, adapter = Path(checkpoint_dir).resolve(), Path(adapter_dir).resolve()
    if ckpt == adapter or ckpt in adapter.parents or adapter in ckpt.parents:
        raise ValueError(
            f"paths.checkpoint_dir ({checkpoint_dir}) and paths.adapter_dir ({adapter_dir}) must be "
            "separate folders, because the checkpoints are deleted once the adapter is saved."
        )
    folders = _checkpoint_folders(checkpoint_dir)
    info = _read_run_info(checkpoint_dir)
    if folders:
        if not resume:
            raise RuntimeError(
                f"{checkpoint_dir} already has checkpoints but training.resume is false. Move or delete "
                "that folder to start from scratch, or set resume: true to carry on."
            )
        if info is None:
            raise RuntimeError(
                f"{checkpoint_dir} has checkpoints but no {RUN_INFO}, so there is no telling which run "
                "they came from. Move or delete that folder to start from scratch."
            )
        diffs = fingerprint_differences(info["fingerprint"], fingerprint)
        if diffs:
            raise RuntimeError(
                f"The checkpoints in {checkpoint_dir} were made with different settings, so they cannot "
                "be resumed:\n  " + "\n  ".join(diffs) + "\nPut the settings back, or move that folder "
                "away to start from scratch."
            )
        for _, p in folders:
            if not is_complete_checkpoint(p):
                print(f"Removing {p}: it was cut off while saving and cannot be resumed.")
                shutil.rmtree(p)
        last = latest_checkpoint(checkpoint_dir)
        if last is not None:
            return last
    if info is None or folders == [] or latest_checkpoint(checkpoint_dir) is None:
        _write_run_info(checkpoint_dir, {
            "fingerprint": fingerprint,
            "train_seconds_so_far": 0.0,
            "jobs": 0,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return None


def record_job_seconds(checkpoint_dir: str, seconds: float) -> float:
    """Add one job's training time to the running total kept next to the checkpoints."""
    info = _read_run_info(checkpoint_dir) or {"fingerprint": {}, "train_seconds_so_far": 0.0, "jobs": 0}
    info["train_seconds_so_far"] = float(info.get("train_seconds_so_far", 0.0)) + seconds
    info["jobs"] = int(info.get("jobs", 0)) + 1
    _write_run_info(checkpoint_dir, info)
    return info["train_seconds_so_far"]


def clear_checkpoints(checkpoint_dir: str) -> None:
    """Delete this run's checkpoints once the adapter is safely saved. Touches nothing else."""
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return
    for _, p in _checkpoint_folders(checkpoint_dir):
        shutil.rmtree(p)
    (d / RUN_INFO).unlink(missing_ok=True)
    try:
        d.rmdir()   # only succeeds if nothing else is in there
    except OSError:
        pass


def guard_adapter_dir(adapter_dir: str, overwrite: bool) -> None:
    """Refuse to replace a finished adapter unless asked to.

    On the cluster outputs/adapter is a link to the group's shared 0.8B adapter in scratch, so
    training through it by mistake would replace everyone's copy.
    """
    if (Path(adapter_dir) / "adapter_config.json").is_file() and not overwrite:
        where = f" (a link to {os.path.realpath(adapter_dir)})" if os.path.islink(adapter_dir) else ""
        raise FileExistsError(
            f"{adapter_dir}{where} already holds a trained adapter. Check that you passed the right "
            "--config. To really replace it, add --overwrite to train_lora.py (or overwrite=True to train())."
        )


def write_training_summary(adapter_dir: str, summary: Dict[str, Any]) -> None:
    path = Path(adapter_dir) / SUMMARY
    old: Dict[str, Any] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    old.update(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2)


def _make_stop_callback():
    """Built on demand so this module still imports without transformers (the light tests need that)."""
    from transformers import TrainerCallback

    class StopOnSignal(TrainerCallback):
        """Save a checkpoint and stop cleanly when Slurm says time is nearly up.

        scripts/violet_train.sbatch asks Slurm for SIGUSR1 twenty minutes before the time limit.
        SIGTERM (scancel, preemption) is handled the same way, though Slurm only waits about 30 s
        after it, so that save may not always make it.
        """

        def __init__(self) -> None:
            self.received: Optional[int] = None
            self._previous: Dict[int, Any] = {}

        def install(self) -> None:
            for name in ("SIGUSR1", "SIGTERM"):
                sig = getattr(signal, name, None)
                if sig is None:
                    continue
                try:
                    self._previous[sig] = signal.signal(sig, self._handle)
                except (ValueError, OSError):   # not the main thread, so leave signals alone
                    pass

        def uninstall(self) -> None:
            for sig, old in self._previous.items():
                try:
                    signal.signal(sig, old)
                except (ValueError, OSError):
                    pass
            self._previous = {}

        def _handle(self, signum, frame) -> None:
            if self.received is None:
                print(f"\nGot signal {signum}: saving a checkpoint after this step, then stopping.", flush=True)
            self.received = signum

        def on_step_end(self, args, state, control, **kwargs):
            if self.received is not None:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.received is not None:
                control.should_evaluate = False   # no time left for a full validation pass
            return control

    return StopOnSignal()


def _qwen_kernel_warning(model) -> None:
    """Qwen3.5's Gated DeltaNet layers are about 10x slower without flash-linear-attention."""
    import importlib.util

    model_type = str(getattr(model.config, "model_type", ""))
    if model_type.startswith("qwen3_5") and importlib.util.find_spec("fla") is None:
        print(
            "WARNING: flash-linear-attention is not installed, so Qwen3.5 runs on the slow reference "
            "kernel (about 10x slower). Install it with: pip install flash-linear-attention==0.5.2",
            flush=True,
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
    train_seconds: float           # all jobs of this run together, when it was resumed
    model_name: str
    adapter_dir: str
    device: str
    precision: str = "fp32"
    per_device_batch_size: int = 0
    grad_accum_steps: int = 0
    global_step: int = 0
    max_steps: int = 0
    stopped_early: bool = False    # a time-limit signal stopped it; run the same command again to resume
    checkpoint: Optional[str] = None
    step_seconds: List[float] = field(default_factory=list)
    peak_memory_gb: Optional[float] = None
    train_losses: List[float] = field(default_factory=list)


def train(
    cfg: Dict[str, Any],
    *,
    overwrite: bool = False,
    max_steps: Optional[int] = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> "TrainResult":
    """Fine-tune the LoRA adapter, save it, and return the model and datasets for evaluation.

    Reply-only loss, slots blanked at random each epoch on train and left filled on val,
    evaluation each epoch, and the adapter saved alone. On top of that, for long cluster jobs:

      * a checkpoint every training.save_steps steps in paths.checkpoint_dir, and the same call
        resumes from the newest one (only if it was made with the same settings);
      * a clean save-and-stop when Slurm signals that the time limit is near (stopped_early=True);
      * on running out of GPU memory, a retry with half the batch and twice the accumulation;
      * the checkpoints are deleted once the final adapter is saved.

    max_steps is for the timing test (scripts/train_lora.py --timing-test): it trains that many
    steps with no checkpoints, no evaluation and nothing saved.
    """
    import gc

    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    from argubot.io import read_jsonl

    timing = max_steps is not None
    t = cfg["training"]
    paths = cfg["paths"]
    adapter_dir = paths["adapter_dir"]
    checkpoint_dir = paths.get("checkpoint_dir") or adapter_dir.rstrip("/\\") + "_checkpoints"
    save_steps = 0 if timing else int(t.get("save_steps") or 0)
    if not timing:
        guard_adapter_dir(adapter_dir, overwrite)   # before anything slow happens

    # Listen for the time-limit signal from the start: loading a big model takes minutes, and an
    # unhandled SIGUSR1 would kill the process outright.
    stopper = _make_stop_callback()
    stopper.install()
    try:
        # Free GPU memory left by a previous (e.g. failed) run in the same kernel.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        set_seed(cfg["seed"])
        cuda = torch.cuda.is_available()
        precision = resolve_precision(t, cuda, cuda and torch.cuda.is_bf16_supported())
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(f"precision: {precision} on {gpu}", flush=True)

        # 1. Base model + tokenizer, falling back only when the config names a fallback.
        model_name = cfg["generator"]["base_model"]
        fallback = cfg["generator"].get("fallback_model")
        try:
            tokenizer, model = _load_base(model_name, dtype)
        except Exception as e:  # noqa: BLE001 - report and fall back rather than crash the run
            if not fallback:
                raise
            print(f"Could not load {model_name!r}: {type(e).__name__}: {e}")
            model_name = fallback
            print(f"Falling back to {model_name!r}. Tell the group.")
            tokenizer, model = _load_base(model_name, dtype)
        if cuda:
            _qwen_kernel_warning(model)

        # GPT-2 names its attention projection c_attn; all-linear is for Qwen3.5.
        if getattr(model.config, "model_type", "") == "gpt2":
            cfg = {**cfg, "lora": {**cfg["lora"], "target_modules": ["c_attn"]}}
            print("GPT-2 detected: using target_modules=['c_attn'].")

        # 2. LoRA. Keep trainable params in fp32 under fp16 or bf16: fp16 needs it to avoid the
        #    "unscale FP16 gradients" error, and bf16 has too few mantissa bits for small LoRA updates.
        model = get_peft_model(model, build_lora_config(cfg))
        if precision != "fp32":
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        model.print_trainable_parameters()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # Gradient checkpointing trades a little compute for a large drop in activation memory.
        # use_cache is incompatible with it, and a frozen base needs inputs to require grad.
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

        # 4. Where to start: a matching checkpoint if there is one, otherwise from scratch.
        resume_from = None
        if save_steps:
            fingerprint = run_fingerprint(cfg, model_name, len(train_examples), len(val_examples), precision)
            resume_from = prepare_checkpoint_dir(checkpoint_dir, adapter_dir, fingerprint, bool(t.get("resume", True)))
            print(f"Resuming from {resume_from}" if resume_from else f"Starting from scratch, checkpoints go to {checkpoint_dir}", flush=True)
        else:
            print("Checkpoints are off for this run (training.save_steps is 0 or this is a timing test).")
        # A copy of the untrained LoRA weights, so an out-of-memory retry with no checkpoint yet
        # really starts from scratch instead of from half-trained weights.
        initial_lora = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}

        # 5. Train, retrying with a smaller batch if the card runs out of memory.
        per_device, accum = int(t["per_device_batch_size"]), int(t["grad_accum_steps"])
        oom_fallback = bool(t.get("oom_fallback", True))
        oom_error = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        timer = _make_step_timer()
        valid = _training_argument_names(TrainingArguments)
        # The timing test saves nothing, so it gets a throwaway folder and never touches the real
        # checkpoint folder of a run that may be half way through.
        trainer_dir = tempfile.mkdtemp(prefix="argubot-timing-") if timing else checkpoint_dir
        start = time.time()
        while True:
            desired = dict(
                output_dir=trainer_dir,
                per_device_train_batch_size=per_device,
                per_device_eval_batch_size=per_device,
                gradient_accumulation_steps=accum,
                learning_rate=float(t["learning_rate"]),
                num_train_epochs=t["num_epochs"],
                warmup_ratio=float(t.get("warmup_ratio") or 0.0),
                fp16=precision == "fp16",
                bf16=precision == "bf16",
                logging_steps=5 if timing else 10,
                save_strategy="steps" if save_steps else "no",
                save_steps=save_steps or 500,          # not used when save_strategy is "no"
                save_total_limit=int(t.get("save_total_limit") or 2),
                report_to="none",
                remove_unused_columns=False,  # our dataset yields dicts the collator consumes as-is
                label_names=["labels"],
                eval_strategy="no" if timing else "epoch",
                seed=cfg["seed"],
            )
            if timing:
                desired["max_steps"] = int(max_steps)
            args = TrainingArguments(**build_training_kwargs(desired, valid))
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=ReplyOnlyCollator(tokenizer.pad_token_id),
                callbacks=[stopper, timer, *(callbacks or [])],
            )
            out_of_memory = False
            try:
                trainer.train(resume_from_checkpoint=resume_from)
            except oom_error:
                if not oom_fallback or per_device <= 1:
                    raise
                out_of_memory = True
            if not out_of_memory:
                break
            # Let go of everything the failed attempt held, then go again with a smaller batch.
            del trainer
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            smaller = halve_batch(per_device, accum)
            print(f"Out of GPU memory at batch {per_device} x {accum}. Retrying at {smaller[0]} x {smaller[1]} "
                  f"(effective batch still {smaller[0] * smaller[1]}).", flush=True)
            per_device, accum = smaller
            timer.reset()
            resume_from = latest_checkpoint(checkpoint_dir) if save_steps else None
            if resume_from is None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in initial_lora:
                            p.copy_(initial_lora[n].to(device=p.device, dtype=p.dtype))
                set_seed(cfg["seed"])
        job_seconds = time.time() - start
        model.config.use_cache = True  # re-enable for fast generation after training

        state = trainer.state
        finished = state.global_step >= state.max_steps
        peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda else None
        device = str(getattr(model, "device", "cpu"))
        result = TrainResult(
            model=model, tokenizer=tokenizer, train_dataset=train_ds, val_dataset=val_ds,
            trainable_params=trainable, total_params=total, train_seconds=job_seconds,
            model_name=model_name, adapter_dir=adapter_dir, device=device, precision=precision,
            per_device_batch_size=per_device, grad_accum_steps=accum,
            global_step=state.global_step, max_steps=state.max_steps,
            step_seconds=list(timer.seconds), peak_memory_gb=peak,
            train_losses=[float(e["loss"]) for e in state.log_history if "loss" in e],
        )
        if timing:
            return result

        # 6a. Stopped by the time limit: leave the checkpoint for the next job.
        if stopper.received is not None and not finished:
            result.stopped_early = True
            result.checkpoint = latest_checkpoint(checkpoint_dir) if save_steps else None
            if save_steps:
                result.train_seconds = record_job_seconds(checkpoint_dir, job_seconds)
            print(f"\nStopped at step {state.global_step} of {state.max_steps}. "
                  + (f"Checkpoint: {result.checkpoint}. Run the same command again to carry on."
                     if result.checkpoint else "No checkpoint was saved (checkpoints are off)."), flush=True)
            return result

        # 6b. Finished: save the adapter only (PEFT writes adapter_config.json + the LoRA weights).
        if save_steps:
            result.train_seconds = record_job_seconds(checkpoint_dir, job_seconds)
            jobs = (_read_run_info(checkpoint_dir) or {}).get("jobs", 1)
        else:
            jobs = 1
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        write_training_summary(adapter_dir, {
            "base_model": model_name,
            "precision": precision,
            "gpu": gpu,
            "per_device_batch_size": per_device,
            "grad_accum_steps": accum,
            "effective_batch": per_device * accum,
            "epochs": t["num_epochs"],
            "warmup_ratio": float(t.get("warmup_ratio") or 0.0),
            "learning_rate": float(t["learning_rate"]),
            "optimizer_steps": state.global_step,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "trainable_params": trainable,
            "total_params": total,
            "train_seconds": round(result.train_seconds, 1),
            "jobs": jobs,
            "peak_memory_gb_last_job": round(peak, 2) if peak is not None else None,
            "eval_loss_per_epoch": [
                {"epoch": round(float(e["epoch"]), 2), "eval_loss": e["eval_loss"]}
                for e in state.log_history if "eval_loss" in e
            ],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if save_steps:
            clear_checkpoints(checkpoint_dir)

        print(f"\nRECORD FOR REPORT -> trainable: {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")
        print(f"RECORD FOR REPORT -> train time: {result.train_seconds:.1f}s over {jobs} job(s) on {gpu}")
        print(f"RECORD FOR REPORT -> precision {precision}, batch {per_device} x {accum}")
        print(f"adapter saved to {adapter_dir} (with {SUMMARY})")
        return result
    finally:
        stopper.uninstall()


def _make_step_timer():
    """Wall time of every optimizer step, for the timing test and the summary."""
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.seconds: List[float] = []
            self._t0: Optional[float] = None

        def reset(self) -> None:
            self.seconds, self._t0 = [], None

        def on_step_begin(self, args, state, control, **kwargs):
            self._t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self._t0 is not None:
                self.seconds.append(time.time() - self._t0)
                self._t0 = None

    return StepTimer()
