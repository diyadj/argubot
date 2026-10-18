"""args.me and IBM-ArgQ loading (owner: Diya).

Written and tested: splitting a long premise into passages that fit the NLI model.
Left for Diya: reading the real files, because the field names must be checked against
the download itself.
"""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Iterator, List

from argubot.schemas import Passage
from argubot.text import split_sentences


def split_into_passages(text: str, max_words: int = 120) -> List[str]:
    """Pack whole sentences into passages of at most max_words words.

    RoBERTa-MNLI reads at most 512 tokens, so a premise of several paragraphs has to be
    cut up first. A single sentence longer than max_words becomes its own passage and the
    scorer truncates it.
    """
    passages: List[str] = []
    current: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        n = len(sentence.split())
        if current and count + n > max_words:
            passages.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        passages.append(" ".join(current))
    return passages


MIN_PASSAGE_WORDS = 8  # args.me has one-liners like "Thanks." that retrieve well on a stray word and say nothing


def _json_array_items(handle, key: str, chunk_chars: int = 1 << 20) -> Iterator[dict]:
    """Stream the objects of the top-level array `key` without loading the whole file.

    args-me-1.0-cleaned.json is 888 MB, which json.load would turn into several GB of Python
    objects. This reads a megabyte at a time and decodes one argument at a time.
    """
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    while True:
        chunk = handle.read(chunk_chars)
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        buf += chunk
        if not started:
            marker = buf.find(f'"{key}"')
            bracket = buf.find("[", marker) if marker >= 0 else -1
            if bracket < 0:
                if not chunk:
                    return
                continue
            buf, started = buf[bracket + 1 :], True
        while True:
            buf = buf.lstrip(" \n\r\t,")
            if not buf or buf[0] == "]":
                break
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break  # the object is cut off at the end of the chunk, read more
            yield item
            buf = buf[end:]
        if not chunk:
            return


def iter_argsme(path: str) -> Iterator[dict]:
    """Arguments from the args.me download: the .zip itself, the .json inside it, or a folder with either."""
    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob("*.zip")) + sorted(p.glob("*.json"))
        if not found:
            raise FileNotFoundError(f"No args.me .zip or .json in {p}. See scripts/download_data.py")
        p = found[0]
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            with z.open(name) as f:
                yield from _json_array_items(f, "arguments")
    else:
        with open(p, "rb") as f:
            yield from _json_array_items(f, "arguments")


def _hard_chunks(text: str, max_words: int) -> List[str]:
    """Cut text with no sentence breaks into max_words pieces (one args.me premise ran 15,000 words)."""
    parts = text.split()
    if len(parts) <= max_words:
        return [text]
    return [" ".join(parts[i : i + max_words]) for i in range(0, len(parts), max_words)]


def load_argsme(path: str, max_words: int = 120) -> List[Passage]:
    """args.me v1.0-cleaned as passages, one or more per premise.

    Field names checked against the real download (Zenodo 4139439, 382,545 arguments):
    {"arguments": [{"id", "conclusion", "premises": [{"text", "stance": "PRO"|"CON"}], "context": {...}}]}.
    The conclusion is the claim the premise supports or attacks, so it becomes Passage.motion and
    the premise stance (PRO or CON) is stance towards it. Premises are cut into passages of at most
    max_words words (the NLI model reads 512 tokens), and passages under MIN_PASSAGE_WORDS words
    are dropped. A piece with no sentence breaks is cut at max_words too. source_id is "<argument id>#<k>", so any passage traces back to its argument.
    """
    passages: List[Passage] = []
    for arg in iter_argsme(path):
        motion = " ".join((arg.get("conclusion") or "").split()) or None
        for premise in arg.get("premises", []):
            stance = premise.get("stance")
            pieces = [
                chunk
                for text in split_into_passages(premise.get("text", ""), max_words)
                for chunk in _hard_chunks(text, max_words)
            ]
            for k, text in enumerate(pieces):
                if len(text.split()) < MIN_PASSAGE_WORDS:
                    continue
                passages.append(
                    Passage(
                        text=text,
                        stance=stance if stance in ("PRO", "CON") else None,
                        source_id=f"{arg.get('id')}#{k}",
                        motion=motion,
                    )
                )
    return passages


_IBM_STANCE = {"1": "PRO", "-1": "CON"}


def load_ibm_argq(path: str) -> List[Passage]:
    """IBM-ArgQ-Rank-30k as passages. `path` is the folder holding train.csv, dev.csv, test.csv.

    Checked against the Hugging Face copy (ibm-research/argument_quality_ranking_30k): columns are
    argument, topic, set, WA, MACE-P, stance_WA (1 or -1), stance_WA_conf. 30,497 arguments on 71
    topics, every one already short (at most 44 words), so nothing needs splitting. The topic is the
    motion and stance_WA is the stance towards it, so the 71 motions double as the retrieval-stance
    test set.
    """
    passages: List[Passage] = []
    for name in ("train", "dev", "test"):
        file = Path(path) / f"{name}.csv"
        with open(file, "r", encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = " ".join(row["argument"].split())
                if not text:
                    continue
                passages.append(
                    Passage(
                        text=text,
                        stance=_IBM_STANCE.get(str(row["stance_WA"]).strip()),
                        source_id=f"ibm-{name}-{i}",
                        motion=row["topic"].strip(),
                    )
                )
    return passages
