"""Dense retriever: MiniLM embeddings with a FAISS index (owner: Diya).

Every passage is encoded once with sentence-transformers (normalized, so inner product equals
cosine) and stored in a flat FAISS index. Search is exact: 1.2M passages x 384 floats is about
1.8 GB in memory, and a query is a single matrix-vector product, so no approximate index is needed.

Encoding is the slow step. A GPU (a free Colab T4) does the 1.2M args.me passages in minutes,
a laptop CPU needs hours, so build once, save, and copy the index folder around. The passages
file is not duplicated inside the index: load() takes the passages in the same order as build().
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from argubot.schemas import Passage

INDEX_FILE = "dense.faiss"
META_FILE = "dense_meta.json"

Encoder = Callable[[Sequence[str]], np.ndarray]


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Optional[Encoder] = None,
        max_seq_length: int = 256,
    ) -> None:
        """encoder replaces the model (tests use a tiny fake one). It maps texts to unit-length rows."""
        self.model_name = model_name
        self.passages: List[Passage] = []
        self._index = None
        self._encoder = encoder
        self._max_seq_length = max_seq_length

    def _encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        if self._encoder is not None:
            return np.asarray(self._encoder(texts), dtype=np.float32)
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self._max_seq_length
        return self._model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

    def build(
        self,
        passages: Sequence[Passage],
        batch_size: int = 256,
        chunk: int = 20_000,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Encode in chunks of `chunk` passages so memory stays flat and progress can be reported."""
        import faiss

        self.passages = list(passages)
        self._index = None
        for start in range(0, len(self.passages), chunk):
            vectors = self._encode([p.text for p in self.passages[start : start + chunk]], batch_size)
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
            if on_progress:
                on_progress(min(start + chunk, len(self.passages)), len(self.passages))

    def save(self, index_dir: str) -> None:
        import faiss

        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(d / INDEX_FILE))
        (d / META_FILE).write_text(
            json.dumps({"model": self.model_name, "n_passages": len(self.passages)}), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        index_dir: str,
        model_name: str,
        passages: Optional[Sequence[Passage]] = None,
        encoder: Optional[Encoder] = None,
    ) -> "DenseRetriever":
        """Load a saved index. `passages` must be the same list, in the same order, that build() saw."""
        import faiss

        d = Path(index_dir)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        if meta["model"] != model_name:
            raise ValueError(f"index was built with {meta['model']}, config asks for {model_name}")
        if passages is None or len(passages) != meta["n_passages"]:
            raise ValueError(
                f"index covers {meta['n_passages']} passages but got "
                f"{'none' if passages is None else len(passages)}; pass the passages file it was built from"
            )
        self = cls(model_name, encoder=encoder)
        self.passages = list(passages)
        self._index = faiss.read_index(str(d / INDEX_FILE))
        return self

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        if self._index is None or not self.passages:
            return []
        scores, ids = self._index.search(self._encode([query]), min(k, len(self.passages)))
        return [replace(self.passages[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
