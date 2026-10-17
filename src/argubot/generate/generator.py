"""Generators (owner: Andrea).

The pipeline only needs the small Generator interface, so the real model, the fallback
model and the test double can be swapped without touching anything else.
"""
from __future__ import annotations

import random
from typing import List, Optional, Protocol, Sequence


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


class HFGenerator:
    """Base model plus the LoRA adapter, sampling with the settings from the config.

    TODO (Andrea), in this order:
      1. Smoke test that the base model loads with your transformers version. Newer releases
         rename torch_dtype to dtype. If Qwen3.5 will not load, use generator.fallback_model.
      2. fp16 on a T4 can give NaNs. If you see them, load in fp32 or use the SCIS GPU.
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
        dtype=None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_prompt_tokens = max_prompt_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # If the prompt is too long, cut the oldest text. Cutting the right side would
        # chop off the [response] tag and the model would continue the wrong thing.
        self.tokenizer.truncation_side = "left"

        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
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
