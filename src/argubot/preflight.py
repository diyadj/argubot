"""Checks that a stage has everything it needs before a GPU job is spent on it.

    python scripts/preflight.py --config configs/config_4b.yaml --stage train

Stages: timing, train, run, eval, demo. Every batch job runs this first and stops if anything
FAILs, so a missing file costs seconds instead of a slot in the queue. It also runs on the login
node, where the GPU checks are skipped (the batch job repeats them on the GPU node).
"""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

STAGES = ("timing", "train", "run", "eval", "demo")
SYSTEMS = ("A", "B", "C1", "C2", "D")


@dataclass
class Check:
    status: str   # OK, WARN, FAIL or SKIP
    what: str
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status:<4}] {self.what}" + (f": {self.detail}" if self.detail else "")


def _exists(path: str) -> bool:
    return Path(path).exists() or Path(path + ".gz").exists()


def _count_lines(path: str) -> int:
    from argubot.io import read_jsonl

    return sum(1 for _ in read_jsonl(path))


def model_in_cache(name: str) -> Optional[bool]:
    """True if the config and the weights index are in the local Hugging Face cache (or a local folder)."""
    local = Path(name)
    if local.is_dir():
        return (local / "config.json").is_file()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(name, filename), str)

    return cached("config.json") and (cached("model.safetensors.index.json") or cached("model.safetensors"))


def detect_gpu() -> Optional[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "bf16": torch.cuda.is_bf16_supported(),
    }


def transcript_progress(transcripts_dir: str) -> Dict[str, Optional[int]]:
    """Debates written per system, or None when the file is missing."""
    out: Dict[str, Optional[int]] = {}
    for s in SYSTEMS:
        path = Path(transcripts_dir) / f"{s}.jsonl"
        out[s] = _count_lines(str(path)) if path.is_file() and path.stat().st_size > 0 else None
    return out


def check_stage(
    cfg: Dict[str, Any],
    stage: str,
    *,
    overwrite: bool = False,
    limit: Optional[int] = None,
    transcripts_dir: Optional[str] = None,
    gpu: Optional[Dict[str, Any]] = None,
    detect: bool = True,
) -> List[Check]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    paths, gen, t = cfg["paths"], cfg["generator"], cfg["training"]
    checks: List[Check] = []
    add = lambda *a: checks.append(Check(*a))  # noqa: E731

    model = gen["base_model"]
    add("OK", "base model", f"{model} (precision {t.get('precision', 'fp16')}, generator dtype {gen.get('dtype', 'auto')})")
    cached = model_in_cache(model)
    if cached is None:
        add("WARN", "model cache", "huggingface_hub is not installed here, so the cache cannot be checked")
    elif cached:
        add("OK", "model cache", f"{model} is downloaded")
    else:
        add("WARN", "model cache", f"{model} is not downloaded yet. Run python scripts/fetch_model.py "
            "--config <this config> on the login node first, or the job will spend time downloading it")

    if stage in ("timing", "train"):
        for key in ("train_examples", "val_examples"):
            ok = _exists(paths[key])
            add("OK" if ok else "FAIL", key, paths[key] + ("" if ok else " is missing (git pull, it is committed as .gz)"))

    if stage == "train":
        adapter = paths["adapter_dir"]
        if (Path(adapter) / "adapter_config.json").is_file():
            add("OK" if overwrite else "FAIL", "adapter folder",
                f"{adapter} already holds an adapter" + (", will be replaced (--overwrite)" if overwrite else
                ". Wrong --config? Add --overwrite only if you really mean to replace it"))
        else:
            add("OK", "adapter folder", f"{adapter} is free")
        checks.extend(_checkpoint_checks(cfg, model))
        out = Path(paths["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(out).free / 1024**3
        add("OK" if free_gb > 5 else "WARN", "disk", f"{free_gb:.0f} GB free where {out} lives (checkpoints need about 2 GB)")

    if stage in ("run", "demo"):
        adapter = paths["adapter_dir"]
        try:
            from argubot.generate.generator import adapter_base_model, check_adapter_matches

            check_adapter_matches(adapter, model)
            add("OK", "adapter", f"{adapter} was trained on {adapter_base_model(adapter)}")
        except (FileNotFoundError, ValueError) as e:
            add("FAIL", "adapter", str(e))
        ok = _exists(paths["passages"])
        add("OK" if ok else "FAIL", "passages", paths["passages"] + ("" if ok else " is missing. Link it in from scratch (docs/DATA_NOTES.md)"))
        method = cfg["retrieval"]["method"]
        index_dir = Path(paths["index_dir"])
        needed = (["bm25_docs.npy", "bm25_weights.npy", "bm25_offsets.npy", "bm25_vocab.json"]
                  if method == "bm25" else ["dense"])
        if cfg["retrieval"].get("use_backup"):
            needed.append("ibm_argq")
        missing = [n for n in needed if not (index_dir / n).exists()]
        add("FAIL" if missing else "OK", f"{method} index",
            f"missing {missing} in {index_dir}. Link them in from scratch (docs/DATA_NOTES.md)" if missing else str(index_dir))
        nli = cfg["nli"]["model"]
        nli_cached = model_in_cache(nli)
        add("OK" if nli_cached else "WARN", "NLI model", nli + ("" if nli_cached else " is not in the cache; the job will download it"))

    if stage == "run":
        ok = _exists(paths["test_dialogues"])
        add("OK" if ok else "FAIL", "test debates", paths["test_dialogues"])
        if ok:
            expected = min(_count_lines(paths["test_dialogues"]), limit) if limit else _count_lines(paths["test_dialogues"])
            tdir = transcripts_dir or paths["transcripts_dir"]
            progress = transcript_progress(tdir)
            done = [s for s, n in progress.items() if n == expected]
            partial = [f"{s} ({n}/{expected})" for s, n in progress.items() if n is not None and n != expected]
            add("OK", "transcripts", f"{tdir}: done {done or 'none'}, still to run "
                f"{[s for s in SYSTEMS if s not in done] or 'none'}")
            if partial:
                add("FAIL", "transcripts", f"wrong number of debates in {partial}. Delete those files so they "
                    "are run again (or use the --limit that made them)")

    if stage == "eval":
        tdir = transcripts_dir or paths["transcripts_dir"]
        progress = transcript_progress(tdir)
        missing = [s for s, n in progress.items() if n is None]
        counts = {n for n in progress.values() if n is not None}
        full = _count_lines(paths["test_dialogues"]) if _exists(paths["test_dialogues"]) else None
        if missing:
            add("FAIL", "transcripts", f"{tdir} has no transcripts for {missing}. Run the systems first")
        elif len(counts) > 1:
            add("FAIL", "transcripts", f"the systems in {tdir} have different numbers of debates: {progress}. "
                "A paired comparison needs the same debates for every system")
        else:
            n = counts.pop()
            if full is not None and n != full and not limit:
                add("WARN", "transcripts", f"{n} debates per system in {tdir}, not the full {full}. "
                    "Fine for a check run, not for the report")
            else:
                add("OK", "transcripts", f"all five systems in {tdir}, {n} debates each")
        strict = cfg["eval"]["strict"]["nli_model"]
        add("OK" if model_in_cache(strict) else "WARN", "strict-metric NLI model", strict)

    checks.extend(_gpu_checks(cfg, stage, gpu if gpu is not None else (detect_gpu() if detect else None)))
    return checks


def _checkpoint_checks(cfg: Dict[str, Any], model: str) -> List[Check]:
    from argubot.generate.training import (
        _read_run_info,
        fingerprint_differences,
        latest_checkpoint,
        run_fingerprint,
    )

    paths, t = cfg["paths"], cfg["training"]
    ckpt = paths.get("checkpoint_dir") or paths["adapter_dir"].rstrip("/\\") + "_checkpoints"
    if not int(t.get("save_steps") or 0):
        return [Check("WARN", "checkpoints", "training.save_steps is 0, so a stopped job starts over")]
    last = latest_checkpoint(ckpt)
    if last is None:
        return [Check("OK", "checkpoints", f"none yet, the run starts from scratch and saves to {ckpt}")]
    info = _read_run_info(ckpt)
    if info is None:
        return [Check("FAIL", "checkpoints", f"{ckpt} has checkpoints from an unknown run. Move that folder away")]
    n_train = _count_lines(paths["train_examples"])
    if t.get("max_train_examples"):
        n_train = min(n_train, int(t["max_train_examples"]))
    now = run_fingerprint(cfg, model, n_train, _count_lines(paths["val_examples"]),
                          info["fingerprint"].get("precision", str(t.get("precision"))))
    diffs = fingerprint_differences(info["fingerprint"], now)
    if diffs:
        return [Check("FAIL", "checkpoints", f"{ckpt} was made with other settings: {'; '.join(diffs)}")]
    hours = float(info.get("train_seconds_so_far", 0)) / 3600
    return [Check("OK", "checkpoints", f"will resume from {Path(last).name} ({hours:.1f} h trained over "
                  f"{info.get('jobs', 0)} job(s) so far)")]


def _gpu_checks(cfg: Dict[str, Any], stage: str, gpu: Optional[Dict[str, Any]]) -> List[Check]:
    if stage == "eval":
        return []
    if gpu is None:
        return [Check("SKIP", "GPU", "torch is not importable here")]
    if not gpu.get("available"):
        return [Check("SKIP", "GPU", "no GPU on this node; the batch job checks again on the GPU node")]
    out = [Check("OK", "GPU", f"{gpu['name']}, {gpu['memory_gb']:.0f} GB")]
    if stage in ("timing", "train") and str(cfg["training"].get("precision")).lower() == "bf16" and not gpu["bf16"]:
        out.append(Check("FAIL", "bf16", f"{gpu['name']} cannot do bf16. Resubmit asking for another card"))
    if "qwen3.5" in cfg["generator"]["base_model"].lower():
        fla = importlib.util.find_spec("fla") is not None
        out.append(Check("OK" if fla else "WARN", "flash-linear-attention",
                         "installed" if fla else "missing, Qwen3.5 will run about 10x slower"))
    return out


def failed(checks: List[Check]) -> bool:
    return any(c.status == "FAIL" for c in checks)
