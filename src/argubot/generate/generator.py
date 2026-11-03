"""Generators (owner: Andrea).

The pipeline only needs the small Generator interface, so the real model, the fallback
model and the test double can be swapped without touching anything else.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence


class Generator(Protocol):
    def generate(self, prompt: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
        """Return n sampled replies for the prompt. Same seed, same samples."""
        ...


class DummyGenerator:
    """Canned debate lines for smoke tests. A few contradict each other on purpose so the
    metrics have something to find. Numbers from this class mean nothing."""

    DEFAULT = (
        "School uniforms reduce bullying in schools.",
        "School uniforms do not reduce bullying in any meaningful way.",
        "Uniforms save families money every year.",
        "Students can express themselves in many ways beyond what they wear.",
        "The evidence linking uniforms to better grades is mixed.",
    )

    def __init__(self, replies: Optional[Sequence[str]] = None) -> None:
        self.replies = list(replies or self.DEFAULT)

    def generate(self, prompt: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
        rng = random.Random(seed)
        return [rng.choice(self.replies) for _ in range(n)]


DTYPE_NAMES = ("auto", "bfloat16", "float16", "float32")


def torch_dtype_from_name(name: Optional[str]) -> Any:
    """generator.dtype from the config as something from_pretrained accepts.

    None and "auto" both mean "the dtype the model ships in", which is the transformers 5.x default.
    """
    if name is None or str(name).lower() == "auto":
        return "auto"
    key = str(name).lower()
    aliases = {"bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32", "float": "float32"}
    key = aliases.get(key, key)
    if key not in DTYPE_NAMES:
        raise ValueError(f"generator.dtype must be one of {DTYPE_NAMES}, got {name!r}")
    import torch

    return getattr(torch, key)


def _model_id_tail(name: str) -> str:
    """'Qwen/Qwen3.5-4B-Base', '/cache/.../Qwen3.5-4B-Base/' and 'Qwen3.5-4B-Base' all give 'qwen3.5-4b-base'."""
    return Path(str(name).rstrip("/\\")).name.lower()


def adapter_base_model(adapter_dir: str) -> Optional[str]:
    """The base model an adapter was trained on, read from its adapter_config.json."""
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f).get("base_model_name_or_path")


def check_adapter_matches(adapter_dir: str, base_model: str) -> None:
    """Stop early with a clear message if the adapter was trained on a different base model.

    PEFT would otherwise fail with a wall of shape-mismatch errors (0.8B and 4B have different
    hidden sizes), or worse, load quietly if two models happened to share shapes.
    """
    if not (Path(adapter_dir) / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter in {adapter_dir} (adapter_config.json is missing). Train it first, or link it "
            "in from scratch (docs/DATA_NOTES.md, 'Trained LoRA adapter on the cluster')."
        )
    trained_on = adapter_base_model(adapter_dir)
    if trained_on and _model_id_tail(trained_on) != _model_id_tail(base_model):
        raise ValueError(
            f"The adapter in {adapter_dir} was trained on {trained_on}, but generator.base_model is "
            f"{base_model}. Use the config that matches the adapter (configs/config.yaml for 0.8B, "
            "configs/config_4b.yaml for 4B)."
        )


class HFGenerator:
    """Base model plus the LoRA adapter, sampling with the settings from the config.

    Notes:
      1. The precision comes from generator.dtype. "auto" loads what the model ships in (bf16 for
         Qwen3.5), which is what transformers 5.x does anyway when no dtype is given.
      2. The adapter has to belong to base_model. That is checked before anything heavy loads.
      3. With a seed, sampling five candidates at once will not reproduce the single sample
         of systems A to C2 exactly. That is fine, but mention it in the report.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: Optional[str] = None,
        *,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_prompt_tokens: int = 1024,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_prompt_tokens = max_prompt_tokens

        if adapter_dir:
            check_adapter_matches(adapter_dir, base_model)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # If the prompt is too long, cut the oldest text. Cutting the right side would
        # chop off the [response] tag and the model would continue the wrong thing.
        self.tokenizer.truncation_side = "left"

        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype_from_name(dtype))
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.to(self.device).eval()

    def generate(self, prompt: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_prompt_tokens
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        return [t.strip() for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
