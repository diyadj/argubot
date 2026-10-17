"""The five systems from the proposal. Each one adds exactly one component to the one before,
so any difference between two neighbours has a single cause."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SystemSpec:
    name: str
    use_retrieval: bool
    use_claims: bool
    use_stance_filter: bool
    use_reranker: bool
    n_candidates: int = 1


def build_systems(n_candidates_rerank: int = 5) -> Dict[str, SystemSpec]:
    return {
        "A": SystemSpec("A", use_retrieval=False, use_claims=False, use_stance_filter=False, use_reranker=False),
        "B": SystemSpec("B", use_retrieval=True, use_claims=False, use_stance_filter=False, use_reranker=False),
        "C1": SystemSpec("C1", use_retrieval=True, use_claims=True, use_stance_filter=False, use_reranker=False),
        "C2": SystemSpec("C2", use_retrieval=True, use_claims=True, use_stance_filter=True, use_reranker=False),
        "D": SystemSpec(
            "D", use_retrieval=True, use_claims=True, use_stance_filter=True, use_reranker=True,
            n_candidates=n_candidates_rerank,
        ),
    }


SYSTEMS = build_systems()
