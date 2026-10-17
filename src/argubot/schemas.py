"""Plain dataclasses that every lane passes around. Change a field here only after telling the group."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    speaker: str  # "user" or "bot"
    text: str


@dataclass
class Dialogue:
    """One scripted debate. User turns are replayed; bot turns (if present) are the real persona replies."""

    dialogue_id: str
    turns: List[Turn]
    split: str = "test"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dialogue":
        return cls(
            dialogue_id=d["dialogue_id"],
            turns=[Turn(**t) for t in d["turns"]],
            split=d.get("split", "test"),
        )


@dataclass
class Passage:
    text: str
    stance: Optional[str] = None  # "PRO", "CON" or None
    score: float = 0.0
    source_id: Optional[str] = None
    motion: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Passage":
        return cls(
            text=d["text"],
            stance=d.get("stance"),
            score=float(d.get("score", 0.0)),
            source_id=d.get("source_id"),
            motion=d.get("motion"),
        )


@dataclass
class NLIScore:
    contradiction: float
    neutral: float
    entailment: float


@dataclass
class TurnRecord:
    """Everything that went into one bot reply, kept so errors can be traced back to a cause."""

    user_text: str
    bot_text: str
    claims_in_prompt: List[str]
    passages_in_prompt: List[str]
    candidates: List[str]
    prompt: str


@dataclass
class Transcript:
    dialogue_id: str
    system: str
    turns: List[TurnRecord] = field(default_factory=list)

    def bot_replies(self) -> List[str]:
        return [t.bot_text for t in self.turns]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transcript":
        return cls(
            dialogue_id=d["dialogue_id"],
            system=d["system"],
            turns=[TurnRecord(**t) for t in d["turns"]],
        )
