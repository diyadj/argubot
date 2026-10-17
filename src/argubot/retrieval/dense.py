"""Dense retriever: MiniLM embeddings with a FAISS index (owner: Diya). Skeleton only.

Suggested plan:
  1. build(): encode every passage with sentence-transformers, normalize_embeddings=True,
     batch in chunks. Encoding 380k passages on a T4 takes a while, so cache the result.
  2. Use faiss.IndexFlatIP (inner product on normalized vectors equals cosine). If search
     is too slow on the full index, try an IVF index.
  3. save() / load(): write the index and the passage ids to cfg["paths"]["index_dir"] so
     nobody has to re-encode before a demo or a rerun.
  4. retrieve(): same signature as BM25Retriever, so scripts can switch with one config value.
"""
from __future__ import annotations

from typing import List, Sequence

from argubot.schemas import Passage


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.passages: List[Passage] = []

    def build(self, passages: Sequence[Passage], batch_size: int = 256) -> None:
        raise NotImplementedError("TODO (Diya): encode passages and build the FAISS index")

    def save(self, index_dir: str) -> None:
        raise NotImplementedError("TODO (Diya)")

    @classmethod
    def load(cls, index_dir: str, model_name: str) -> "DenseRetriever":
        raise NotImplementedError("TODO (Diya)")

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        raise NotImplementedError("TODO (Diya)")
