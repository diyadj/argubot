"""The turn loop shared by all five systems (owner: Andrea).

One class runs every system. A SystemSpec switches components on, and nothing else differs,
which is what keeps the ablation fair. Decoding settings come from the config, and each
turn gets a seed derived from the debate and the turn number, so every system faces the
same pushback with the same sampling luck wherever it can.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable, Dict, List, Optional

from argubot.claims.tracker import ClaimTracker
from argubot.generate.generator import Generator
from argubot.nli.rerank import rerank
from argubot.nli.scorer import NLIScorer
from argubot.prompt import build_prompt
from argubot.retrieval.base import Retriever, build_query
from argubot.retrieval.stance_filter import StanceConsistencyFilter
from argubot.schemas import Dialogue, Passage, Transcript, Turn, TurnRecord
from argubot.systems.specs import SystemSpec


def turn_seed(base_seed: int, dialogue_id: str, turn_index: int) -> int:
    return zlib.crc32(f"{base_seed}:{dialogue_id}:{turn_index}".encode()) % (2**31)


class ArguBotPipeline:
    def __init__(
        self,
        spec: SystemSpec,
        generator: Generator,
        cfg: Dict[str, Any],
        retriever: Optional[Retriever] = None,
        nli: Optional[NLIScorer] = None,
        tracker_factory: Optional[Callable[[], ClaimTracker]] = None,
    ) -> None:
        if spec.use_retrieval and retriever is None:
            raise ValueError(f"System {spec.name} needs a retriever")
        if (spec.use_stance_filter or spec.use_reranker) and nli is None:
            raise ValueError(f"System {spec.name} needs an NLI scorer")

        self.spec = spec
        self.generator = generator
        self.cfg = cfg
        self.retriever = retriever
        self.nli = nli
        self.tracker_factory = tracker_factory or (
            lambda: ClaimTracker(
                max_claims=cfg["claims"]["max_claims"],
                dedup_threshold=cfg["claims"]["dedup_threshold"],
            )
        )
        self.stance_filter = (
            StanceConsistencyFilter(nli, cfg["nli"]["contradiction_threshold"])
            if spec.use_stance_filter
            else None
        )

    def _passages_for(self, user_text: str, claims: List[str]) -> List[Passage]:
        if not self.spec.use_retrieval:
            return []
        r = self.cfg["retrieval"]
        query = build_query(user_text, claims)
        passages = self.retriever.retrieve(query, k=r["top_k_retrieve"])
        if self.stance_filter is not None:
            # Filter the whole top 10 first, then keep the top 3 that survive.
            passages = self.stance_filter.filter(passages, claims)
        return passages[: r["top_k_keep"]]

    def _choose(self, candidates: List[str], claims: List[str]) -> str:
        if self.spec.use_reranker and len(candidates) > 1:
            best, _ = rerank(candidates, claims, self.nli, min_words=self.cfg["nli"]["min_reply_words"])
            return candidates[best]
        return candidates[0]

    def step(
        self,
        history: List[Turn],
        tracker: ClaimTracker,
        user_turn: Turn,
        dialogue_id: str,
        turn_index: int,
    ) -> TurnRecord:
        """Answer one user turn. Appends both turns to history and feeds the reply to the tracker.

        run_debate and the live chat app both go through here, so the demo behaves
        exactly like the systems that were evaluated.
        """
        p = self.cfg["prompt"]
        history.append(user_turn)
        # The tracker always listens to the bot so the metrics can use it, but only
        # systems C1 to D are allowed to show the claim log to the generator.
        claims = list(tracker.claims) if self.spec.use_claims else []
        passages = self._passages_for(user_turn.text, claims)
        prompt = build_prompt(
            history,
            claims,
            [ps.text for ps in passages],
            max_history_turns=p["max_history_turns"],
            max_passage_words=p["max_passage_words"],
        )
        seed = turn_seed(self.cfg["seed"], dialogue_id, turn_index)
        candidates = self.generator.generate(prompt, n=self.spec.n_candidates, seed=seed)
        reply = self._choose(candidates, claims)

        history.append(Turn("bot", reply))
        tracker.update(reply)
        return TurnRecord(
            user_text=user_turn.text,
            bot_text=reply,
            claims_in_prompt=claims,
            passages_in_prompt=[ps.text for ps in passages],
            candidates=list(candidates),
            prompt=prompt,
        )

    def run_debate(self, dialogue: Dialogue) -> Transcript:
        """Replay the scripted user turns and let the system answer each one."""
        tracker = self.tracker_factory()
        history: List[Turn] = []
        records: List[TurnRecord] = []
        user_turns = [t for t in dialogue.turns if t.speaker == "user"]
        for i, user_turn in enumerate(user_turns):
            records.append(self.step(history, tracker, user_turn, dialogue.dialogue_id, i))
        return Transcript(dialogue_id=dialogue.dialogue_id, system=self.spec.name, turns=records)
