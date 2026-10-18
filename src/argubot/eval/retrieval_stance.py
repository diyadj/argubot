"""Retrieval-stance test on the 71 IBM-ArgQ motions (owner: Diya).

Question: does the retriever hand back passages that take the same side as the query? A keyword
or embedding match sees topic words, not sides, so a user's argument about "school uniforms" can
retrieve arguments for and against uniforms alike. The stance filter (system C2) exists because of
that, and this test measures how much of it there is to fix.

The corpus is IBM-ArgQ, where every argument has a motion (the topic) and a stance towards it
(PRO or CON), so the answers are known. The definitions below are ours, because the proposal's
exact wording was not available when this was written; change them here if it differs.

  Test 1, motion as query. Query = the motion text. Metrics: motion hit@k (share of the top k
  that belong to that motion) and PRO share among those hits (0.5 means balanced).

  Test 2, argument as query. Query = one argument (itself removed from the results). Metrics:
    topical hit@k   share of the top k from the query's own motion
    same-stance     among the top-k results from the query's own motion, the share with the
                    query's stance
    chance          the same-stance share a random draw from that motion would give,
                    (same-stance arguments - 1) / (arguments in the motion - 1)
  A retriever that understands sides beats chance. One that only matches topics sits on it.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from argubot.eval.stats import paired_bootstrap_ci
from argubot.retrieval.base import Retriever
from argubot.schemas import Passage


def motion_query_test(retriever: Retriever, passages: Sequence[Passage], k: int = 10) -> List[Dict[str, Any]]:
    """One row per motion: how many of the top k come from it, and how balanced they are."""
    rows = []
    for motion in sorted({p.motion for p in passages if p.motion}):
        got = retriever.retrieve(motion, k=k)
        same = [r for r in got if r.motion == motion]
        rows.append(
            {
                "motion": motion,
                "hit_at_k": len(same) / k if got else 0.0,
                "pro_share": (sum(r.stance == "PRO" for r in same) / len(same)) if same else None,
                "n_hits": len(same),
            }
        )
    return rows


def argument_query_test(
    retriever: Retriever,
    passages: Sequence[Passage],
    k: int = 10,
    max_queries: Optional[int] = None,
    seed: int = 42,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Leave-one-out stance test, with a chance baseline and a paired bootstrap CI for the gap."""
    per_motion = Counter(p.motion for p in passages)
    per_motion_stance = Counter((p.motion, p.stance) for p in passages)

    queries = [p for p in passages if p.motion and p.stance and per_motion[p.motion] > 1]
    if max_queries and len(queries) > max_queries:
        queries = random.Random(seed).sample(queries, max_queries)

    topical, same_rate, chance = [], [], []
    for q in queries:
        got = [r for r in retriever.retrieve(q.text, k=k + 1) if r.source_id != q.source_id][:k]
        if not got:
            continue
        same_motion = [r for r in got if r.motion == q.motion]
        topical.append(len(same_motion) / len(got))
        if same_motion:
            same_rate.append(sum(r.stance == q.stance for r in same_motion) / len(same_motion))
            chance.append((per_motion_stance[(q.motion, q.stance)] - 1) / (per_motion[q.motion] - 1))

    ci = paired_bootstrap_ci(same_rate, chance, n_boot=n_boot, seed=seed)
    return {
        "k": k,
        "n_queries": len(topical),
        "n_queries_with_a_same_motion_result": len(same_rate),
        "topical_hit_at_k": float(np.mean(topical)),
        "same_stance_rate": float(np.mean(same_rate)),
        "chance_same_stance_rate": float(np.mean(chance)),
        "gap_over_chance": ci.mean_diff,
        "gap_ci_low": ci.ci_low,
        "gap_ci_high": ci.ci_high,
    }
