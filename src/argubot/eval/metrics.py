"""Automatic metrics (owner: Cassey).

Contradiction rate is the metric the whole hypothesis rests on (C1 versus B), so it is
core, not an extension. Perplexity, distinct-2 and claims per turn are the guardrails that
stop a system from looking consistent by being generic, repetitive or claim-free.

Two versions of the contradiction rate share one function. The original (the proposal's) flags a
turn if any reply sentence contradicts any earlier claim by RoBERTa-MNLI in either direction. The
strict version exists because the original flags real human CMV replies about as often as the
bots, and reading 40 of those flags found no real contradiction: most came from "claims" that are
questions, orders or the opponent's view, from fragments like "Rights do not.", and from NLI calling
unrelated sentences contradictory. The strict version only compares real assertions with
self-contained sentences on the same subject, asks both NLI directions to agree, and uses a
different NLI model from the one system D reranks with, so D is not graded by its own judge.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from argubot.claims.tracker import extract_claims
from argubot.nli.scorer import NLIScorer, symmetric_contradiction_batch
from argubot.schemas import Dialogue, Transcript, TurnRecord
from argubot.text import split_sentences

PairScorer = Callable[[Sequence[Tuple[str, str]]], List[float]]


@dataclass
class ContradictionResult:
    rate: float
    n_turns: int
    n_flagged: int
    per_dialogue: Dict[str, float] = field(default_factory=dict)
    flagged: List[dict] = field(default_factory=list)  # pairs for the raters' spot check
    n_sentences: int = 0  # reply sentences on scored turns, all of them, filtered or not
    n_flagged_sentences: int = 0
    per_dialogue_sentence: Dict[str, float] = field(default_factory=dict)

    @property
    def sentence_rate(self) -> float:
        """Share of reply sentences that contradict an earlier claim: fairer to short replies."""
        return self.n_flagged_sentences / self.n_sentences if self.n_sentences else 0.0


def contradiction_rate(
    transcripts: Sequence[Transcript],
    scorer: NLIScorer,
    threshold: float = 0.5,
    extract: Optional[Callable[[str], List[str]]] = None,
    *,
    keep_claim: Optional[Callable[[str], bool]] = None,
    keep_sentence: Optional[Callable[[str], bool]] = None,
    pair_scores: Optional[PairScorer] = None,
) -> ContradictionResult:
    """Share of bot turns after the first where a reply sentence contradicts an earlier bot claim.

    Both NLI directions are checked and the larger score is used. The earlier claims are NOT
    capped at the tracker's five, so the metric sees more than the system does.

    The default extractor is the same one the claim tracker uses. Pass extract=split_sentences
    to compare against every earlier sentence instead. That is the version to use for the
    independent checker in the proposal's extension, since a tracker mistake then cannot hide a
    contradiction from both the system and the metric.

    keep_claim, keep_sentence and pair_scores turn this into the strict metric (strict_options);
    left out, the result is the original metric exactly.
    """
    extract = extract or extract_claims
    pair_scores = pair_scores or (lambda pairs: symmetric_contradiction_batch(scorer, pairs))
    n_turns = n_flagged = n_sentences = n_flagged_sentences = 0
    per_dialogue: Dict[str, float] = {}
    per_dialogue_sentence: Dict[str, float] = {}
    flagged: List[dict] = []

    for t in transcripts:
        earlier: List[str] = []
        d_turns = d_flagged = d_sentences = d_flagged_sentences = 0
        for i, rec in enumerate(t.turns):
            sentences = split_sentences(rec.bot_text)
            if i > 0:
                d_turns += 1
                d_sentences += len(sentences)
                checked = [s for s in sentences if keep_sentence is None or keep_sentence(s)]
                pairs = [(claim, sent) for claim in earlier for sent in checked]
                scores = pair_scores(pairs)
                hits = [(p, s) for p, s in zip(pairs, scores) if s >= threshold]
                if hits:
                    d_flagged += 1
                    d_flagged_sentences += len({sent for (_, sent), _ in hits})
                    for (claim, sent), s in hits:
                        flagged.append(
                            {
                                "dialogue_id": t.dialogue_id,
                                "system": t.system,
                                "turn": i,
                                "earlier_claim": claim,
                                "new_sentence": sent,
                                "score": s,
                            }
                        )
            earlier.extend(c for c in extract(rec.bot_text) if keep_claim is None or keep_claim(c))
        n_turns += d_turns
        n_flagged += d_flagged
        n_sentences += d_sentences
        n_flagged_sentences += d_flagged_sentences
        per_dialogue[t.dialogue_id] = (d_flagged / d_turns) if d_turns else 0.0
        per_dialogue_sentence[t.dialogue_id] = (d_flagged_sentences / d_sentences) if d_sentences else 0.0

    rate = n_flagged / n_turns if n_turns else 0.0
    return ContradictionResult(
        rate, n_turns, n_flagged, per_dialogue, flagged, n_sentences, n_flagged_sentences, per_dialogue_sentence
    )


# --- The strict metric -------------------------------------------------------------------------

_QUOTED_OPPONENT = re.compile(
    r"\b(you|you're|you are|you've)\s+(say|said|saying|claim|claimed|claiming|seem|think|argue|arguing|"
    r"mean|suggest|suggesting|believe|imply|implying)\b|\byour (argument|point|view|claim|position)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL = re.compile(r"^(if|suppose|imagine|assuming|assume|say|let's say|what if|unless)\b", re.IGNORECASE)
_IMPERATIVE_START = {
    "don't", "do", "let", "let's", "be", "see", "look", "imagine", "consider", "think", "try", "stop",
    "go", "take", "remember", "read", "tell", "ask", "note", "keep", "give", "check", "please",
}
_DANGLING_START = {
    "it", "it's", "they", "they're", "that", "that's", "this", "these", "those", "he", "she", "him",
    "her", "them", "so", "and", "but", "or", "yes", "no", "yeah", "nope", "which", "because", "then",
    "also", "still", "same", "exactly", "indeed",
}


def _first_word(sentence: str) -> str:
    words = re.findall(r"[a-z']+", sentence.lower().replace("’", "'"))
    return words[0] if words else ""


def is_assertion(claim: str) -> bool:
    """False for "claims" that are not the speaker's own statement: questions, orders,
    hypotheticals and restatements of the opponent ("All you're saying is ...")."""
    text = claim.strip().replace("’", "'")
    return bool(text) and not (
        text.endswith("?")
        or _first_word(text) in _IMPERATIVE_START
        or _HYPOTHETICAL.match(text)
        or _QUOTED_OPPONENT.search(text)
    )


def is_standalone(sentence: str, min_words: int = 6) -> bool:
    """False for sentences that cannot be judged without the conversation around them:
    questions, fragments ("Rights do not.") and openings that point back ("It is.", "So ...")."""
    text = sentence.strip()
    return len(text.split()) >= min_words and not text.endswith("?") and _first_word(text) not in _DANGLING_START


def word_overlap_similarity(pairs: Sequence[Tuple[str, str]]) -> List[float]:
    """Word-set Jaccard. A model-free stand-in for EmbeddingSimilarity in tests and --dummy runs."""
    out = []
    for a, b in pairs:
        wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
        out.append(len(wa & wb) / len(wa | wb) if wa | wb else 0.0)
    return out


class EmbeddingSimilarity:
    """Cosine similarity of sentence embeddings (sentence-transformers), cached per sentence."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, object] = {}

    def __call__(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        new = list({s for pair in pairs for s in pair if s not in self._cache})
        if new:
            for s, v in zip(new, self.model.encode(new, normalize_embeddings=True, batch_size=64)):
                self._cache[s] = v
        return [float(self._cache[a] @ self._cache[b]) for a, b in pairs]


def strict_pair_scores(scorer: NLIScorer, similarity: PairScorer, min_similarity: float) -> PairScorer:
    """Contradiction score that is 0 for pairs on different subjects and otherwise the smaller
    of the two NLI directions, so both have to agree. Only same-subject pairs reach the NLI model."""

    def score(pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        sims = similarity(pairs)
        related = [i for i, s in enumerate(sims) if s >= min_similarity]
        out = [0.0] * len(pairs)
        for i, s in zip(related, symmetric_contradiction_batch(scorer, [pairs[i] for i in related], combine=min)):
            out[i] = s
        return out

    return score


def strict_options(scorer: NLIScorer, similarity: PairScorer, min_similarity: float, min_words: int) -> dict:
    """Keyword arguments that make contradiction_rate the strict metric."""
    return {
        "keep_claim": is_assertion,
        "keep_sentence": lambda s: is_standalone(s, min_words),
        "pair_scores": strict_pair_scores(scorer, similarity, min_similarity),
    }


def human_transcripts(dialogues: Sequence[Dialogue]) -> List[Transcript]:
    """The real CMV persona replies of the test debates, as transcripts of a system called "human".

    Scored like any system, they give the metric's noise floor: people rarely contradict themselves
    outright, so whatever rate they get is mostly NLI and claim-extraction error. Dialogues
    without bot turns (the toy ones) give empty transcripts.
    """
    out = []
    for d in dialogues:
        turns, user = [], ""
        for t in d.turns:
            if t.speaker == "user":
                user = t.text
            else:
                turns.append(TurnRecord(user, t.text, [], [], [], ""))
        out.append(Transcript(d.dialogue_id, "human", turns))
    return out


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Unique n-grams divided by all n-grams. Low values mean repetitive replies."""
    grams: List[tuple] = []
    for text in texts:
        tokens = text.split()
        grams.extend(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / len(grams) if grams else 0.0


def claims_per_turn(
    transcripts: Sequence[Transcript], extract: Optional[Callable[[str], List[str]]] = None
) -> float:
    extract = extract or extract_claims
    counts = [len(extract(rec.bot_text)) for t in transcripts for rec in t.turns]
    return sum(counts) / len(counts) if counts else 0.0


def mean_reply_words(transcripts: Sequence[Transcript]) -> float:
    lengths = [len(rec.bot_text.split()) for t in transcripts for rec in t.turns]
    return sum(lengths) / len(lengths) if lengths else 0.0


def perplexity(model, dataloader, ignore_index: int = -100) -> float:
    """Token-weighted perplexity over reply tokens only.

    The base notebook counted pad tokens and the prompt. Here only labels that are not
    ignore_index count, so batches of different lengths are weighted correctly.
    Evaluate with full slots (system C conditions) and note it in the report.
    """
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            n_tokens = int((batch["labels"][:, 1:] != ignore_index).sum().item())
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
    return math.exp(total_nll / max(total_tokens, 1))
