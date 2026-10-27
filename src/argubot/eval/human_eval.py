"""Blinded human evaluation (owner: Cassey builds it, Lumen runs the logistics).

Every rater gets the same dialogues for every system, shuffled into a different order,
with no system name anywhere on the sheet. A private key file maps item ids back to systems.
Each criterion is scored 0 or 1. The proposal does not fix a scale, so confirm with the group
before the sheets go out. If you switch to a 1 to 5 scale, use a weighted kappa instead.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from argubot.eval.stats import cohens_kappa
from argubot.schemas import Transcript

CRITERIA = ["kept_position", "answered_counterargument", "used_support_correctly"]


def render_dialogue(transcript: Transcript) -> str:
    lines: List[str] = []
    for rec in transcript.turns:
        lines.append(f"User: {rec.user_text}")
        lines.append(f"Bot: {rec.bot_text}")
    return "\n".join(lines)


def make_rating_sheets(
    transcripts_by_system: Dict[str, Sequence[Transcript]],
    out_dir: str,
    n_per_system: int = 20,
    n_raters: int = 2,
    seed: int = 0,
) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shared = None
    for ts in transcripts_by_system.values():
        ids = {t.dialogue_id for t in ts}
        shared = ids if shared is None else shared & ids
    chosen = random.Random(seed).sample(sorted(shared or []), min(n_per_system, len(shared or [])))

    items = []  # (item_id, system, transcript)
    for system, ts in transcripts_by_system.items():
        by_id = {t.dialogue_id: t for t in ts}
        for did in chosen:
            items.append((system, by_id[did]))
    random.Random(seed).shuffle(items)
    items = [(f"item{i:03d}", s, t) for i, (s, t) in enumerate(items)]

    key_path = out / "key_PRIVATE.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "system", "dialogue_id"])
        for item_id, system, t in items:
            w.writerow([item_id, system, t.dialogue_id])

    paths = [key_path]
    for r in range(n_raters):
        order = list(items)
        random.Random(seed + 1 + r).shuffle(order)
        path = out / f"rater_{r + 1}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "dialogue", *CRITERIA, "notes"])
            for item_id, _, t in order:
                w.writerow([item_id, render_dialogue(t), *[""] * len(CRITERIA), ""])
        paths.append(path)
    return paths


def _read(path: str) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["item_id"]: row for row in csv.DictReader(f)}


def kappa_per_criterion(rater1_csv: str, rater2_csv: str) -> Dict[str, Optional[float]]:
    """Cohen's kappa for each criterion, using only items both raters have filled in."""
    a, b = _read(rater1_csv), _read(rater2_csv)
    result: Dict[str, Optional[float]] = {}
    for crit in CRITERIA:
        pairs = [
            (a[i][crit].strip(), b[i][crit].strip())  # "1 " and "1" are the same score
            for i in a
            if i in b and a[i][crit].strip() != "" and b[i][crit].strip() != ""
        ]
        result[crit] = cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
    return result


def check_sheet(path: str, items: Sequence[str]) -> List[str]:
    """Problems that would quietly skew the scores: missing items, blanks, anything but 0 or 1."""
    rows = _read(path)
    problems = [f"{i}: missing from the sheet" for i in items if i not in rows]
    for item_id, row in rows.items():
        for crit in CRITERIA:
            value = row[crit].strip()
            if value == "":
                problems.append(f"{item_id} {crit}: blank")
            elif value not in ("0", "1"):
                problems.append(f"{item_id} {crit}: {row[crit]!r} is not 0 or 1")
    return problems


def mean_scores_by_system(key_csv: str, rater_csvs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Average score per system and criterion across all raters and filled-in items."""
    key = _read(key_csv)
    sums: Dict[str, Dict[str, List[float]]] = {}
    for path in rater_csvs:
        for item_id, row in _read(path).items():
            system = key[item_id]["system"]
            for crit in CRITERIA:
                if row[crit].strip() != "":
                    sums.setdefault(system, {}).setdefault(crit, []).append(float(row[crit]))
    return {s: {c: sum(v) / len(v) for c, v in crits.items()} for s, crits in sums.items()}
