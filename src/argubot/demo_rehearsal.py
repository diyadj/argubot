"""Rehearse the live demo: scripted debates, run through the demo's own code path, scored for a live audience.

scripts/demo_rehearsal.py plays each flow below several times (different sampling seeds) on several
systems and ranks the flow and system pairs that are most likely to go well in front of people. The
scoring here is plain arithmetic so it can be tested without a model.

What makes a run "demo ready":
  * the strict metric flags no contradiction in it (its flags are often false alarms, see
    docs/RESULTS.md, so a flagged run is still worth reading before it is ruled out);
  * every reply is a normal length (not a fragment, not a wall of text);
  * for systems with retrieval, at least half of the passages shown are on the debate's topic.
On top of that, ranking prefers runs where the last reply comes back to the bot's first point, which is
what the audience should see the claim log doing, and faster replies.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Each flow: the user argues one side; the bot (trained on CMV commenters who challenge the poster)
# argues the other. The last turn is a bait that invites the bot to concede its own earlier point.
FLOWS: List[Dict] = [
    {"id": "uniforms", "title": "School uniforms", "topic": ["uniform"], "turns": [
        "School uniforms should be banned. They stop students from expressing who they are.",
        "But uniforms are expensive for families, and kids outgrow them every year.",
        "So you agree that uniforms don't actually do anything about bullying?"]},
    {"id": "death-penalty", "title": "The death penalty", "topic": ["death penalty", "capital punishment", "execut"], "turns": [
        "The death penalty should be abolished. The state should never be allowed to kill its own citizens.",
        "Innocent people have been executed, and you can never undo that.",
        "So you agree the death penalty doesn't really deter crime?"]},
    {"id": "social-media", "title": "Social media for under-16s", "topic": ["social media", "facebook", "instagram", "teen", "online"], "turns": [
        "Children under 16 should be banned from social media. It is wrecking their mental health.",
        "Teenagers can't handle constantly comparing themselves to influencers.",
        "So you'd agree social media does more harm than good for teenagers?"]},
    {"id": "minimum-wage", "title": "Raising the minimum wage", "topic": ["minimum wage", "wage"], "turns": [
        "We should raise the minimum wage to 20 dollars an hour. Nobody working full time should live in poverty.",
        "Higher wages mean workers spend more, which helps local businesses.",
        "So you agree that raising the minimum wage won't really cost jobs?"]},
    {"id": "nuclear", "title": "Nuclear power", "topic": ["nuclear"], "turns": [
        "Nuclear power is too dangerous. We should shut down every nuclear plant.",
        "Look at Chernobyl and Fukushima. One accident can ruin a region for decades.",
        "So you agree nuclear waste is a problem nobody has solved?"]},
    {"id": "animal-testing", "title": "Animal testing", "topic": ["animal"], "turns": [
        "Animal testing should be completely banned. It is cruel and unnecessary.",
        "We now have computer models and cell cultures that can replace animals.",
        "So you agree that animals suffer needlessly in most experiments?"]},
    {"id": "homework", "title": "Homework", "topic": ["homework"], "turns": [
        "Homework should be abolished. Kids already spend enough hours at school.",
        "Studies show homework barely improves results for younger students.",
        "So you agree homework mostly just causes stress?"]},
    {"id": "zoos", "title": "Zoos", "topic": ["zoo"], "turns": [
        "Zoos should be closed. Keeping wild animals in cages is cruel.",
        "Animals in zoos often show signs of stress, like pacing back and forth.",
        "So you agree zoos are mainly about entertainment, not conservation?"]},
]

BAIT_TURN = 2   # index of the bait turn in every flow
MIN_WORDS, MAX_WORDS = 20, 170
_STOP = set("""a an the and or but if so of to in on at for with by from as is are was were be been being it its
this that these those i you he she we they them their our your my me us not no do does did have has had
than then there here what which who whom why how all any some more most such can could would should will
just very also about into over only own same too out up down off again further once really""".split())


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP and len(w) > 2}


def word_count(text: str) -> int:
    return len(text.split())


def on_topic_share(passages_per_turn: Sequence[Sequence[str]], topic: Sequence[str]) -> Optional[float]:
    """Share of all passages shown that mention the topic; None when the system retrieves nothing."""
    shown = [p.lower() for turn in passages_per_turn for p in turn]
    if not shown:
        return None
    return sum(1 for p in shown if any(t in p for t in topic)) / len(shown)


def callback_score(first_claims: Sequence[str], last_reply: str) -> float:
    """How strongly the last reply comes back to a claim made in the first reply (0 to 1).

    The share of a first-turn claim's content words that reappear in the last reply, best claim taken.
    """
    last = content_words(last_reply)
    best = 0.0
    for claim in first_claims:
        words = content_words(claim)
        if words:
            best = max(best, len(words & last) / len(words))
    return best


@dataclass
class TrialScore:
    flow: str
    system: str
    trial: int
    flagged_turns: List[int]
    replies_ok: bool
    on_topic: Optional[float]
    callback: float
    claims_after_first: int
    seconds: List[float] = field(default_factory=list)

    @property
    def bait_clean(self) -> bool:
        return BAIT_TURN not in self.flagged_turns

    @property
    def ready(self) -> bool:
        topic_ok = self.on_topic is None or self.on_topic >= 0.5
        return not self.flagged_turns and self.replies_ok and topic_ok

    @property
    def mean_seconds(self) -> float:
        return statistics.mean(self.seconds) if self.seconds else 0.0


def score_trial(
    flow: Dict, system: str, trial: int, replies: Sequence[str], passages_per_turn: Sequence[Sequence[str]],
    claims_after_each_turn: Sequence[Sequence[str]], flagged_turns: Sequence[int], seconds: Sequence[float],
) -> TrialScore:
    return TrialScore(
        flow=flow["id"],
        system=system,
        trial=trial,
        flagged_turns=sorted(set(int(t) for t in flagged_turns)),
        replies_ok=all(MIN_WORDS <= word_count(r) <= MAX_WORDS for r in replies),
        on_topic=on_topic_share(passages_per_turn, flow["topic"]),
        callback=callback_score(claims_after_each_turn[0] if claims_after_each_turn else [], replies[-1] if replies else ""),
        claims_after_first=len(claims_after_each_turn[0]) if claims_after_each_turn else 0,
        seconds=list(seconds),
    )


@dataclass
class ComboSummary:
    flow: str
    system: str
    trials: int
    ready: int
    bait_clean: int
    callback: float
    on_topic: Optional[float]
    claims_after_first: float
    seconds: float

    def sort_key(self):
        return (-self.ready, -self.bait_clean, -round(self.callback, 2),
                -(self.on_topic if self.on_topic is not None else 0.0), self.seconds)


def summarize(scores: Sequence[TrialScore]) -> List[ComboSummary]:
    groups: Dict[tuple, List[TrialScore]] = {}
    for s in scores:
        groups.setdefault((s.flow, s.system), []).append(s)
    out = []
    for (flow, system), ts in groups.items():
        topics = [t.on_topic for t in ts if t.on_topic is not None]
        out.append(ComboSummary(
            flow=flow, system=system, trials=len(ts),
            ready=sum(t.ready for t in ts),
            bait_clean=sum(t.bait_clean for t in ts),
            callback=statistics.mean(t.callback for t in ts),
            on_topic=statistics.mean(topics) if topics else None,
            claims_after_first=statistics.mean(t.claims_after_first for t in ts),
            seconds=statistics.mean(t.mean_seconds for t in ts),
        ))
    return out


def rank(summaries: Sequence[ComboSummary], exclude_systems: Sequence[str] = ("A",)) -> List[ComboSummary]:
    """Best flow and system pairs for the main demo, most demo-ready first."""
    return sorted((s for s in summaries if s.system not in exclude_systems), key=ComboSummary.sort_key)


def best_trial(scores: Sequence[TrialScore], flow: str, system: str) -> Optional[TrialScore]:
    """The run to show for a pair: a ready one with the strongest callback, else the fewest flags."""
    ts = [s for s in scores if s.flow == flow and s.system == system]
    if not ts:
        return None
    return sorted(ts, key=lambda t: (not t.ready, len(t.flagged_turns), -t.callback, t.mean_seconds))[0]


def contrast_candidates(summaries: Sequence[ComboSummary], scores: Sequence[TrialScore],
                        memory_systems: Sequence[str] = ("C1", "C2", "D")) -> List[str]:
    """Flows where system A was flagged at least once and some system with a claim log never was."""
    out = []
    for flow in sorted({s.flow for s in summaries}):
        a_flagged = any(t.flagged_turns for t in scores if t.flow == flow and t.system == "A")
        clean = [s.system for s in summaries if s.flow == flow and s.system in memory_systems
                 and all(not t.flagged_turns for t in scores if t.flow == flow and t.system == s.system)]
        if a_flagged and clean:
            out.append(flow)
    return out
