"""Retrievers pick the labelled examples shown to the LLM: the k records most similar to the query.

TfidfRetriever runs anywhere. SbertRetriever uses sentence embeddings, as the 2023 work did after Liu et al. (2021),
and needs the optional `sentence-transformers` package. Both only return records from the query's 3-digit sector,
because the prompt is built per sector.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import Record


class Retriever(Protocol):
    def top_k(self, record: Record, k: int) -> list[Record]:
        """Up to k labelled records from the same sector as `record`, most similar first, never `record` itself."""
        ...


class _SimilarityRetriever:
    """Shared ranking: subclasses supply `_embed(texts) -> L2-normalised matrix`."""

    def __init__(self, pool: list[Record]):
        self._pool = list(pool)
        self._pool_vectors = self._embed_pool([r.text for r in self._pool])

    def _embed_pool(self, texts):
        raise NotImplementedError

    def _embed_query(self, text):
        raise NotImplementedError

    def top_k(self, record: Record, k: int) -> list[Record]:
        if not self._pool or k <= 0:
            return []
        scores = self._pool_vectors @ self._embed_query(record.text).T
        scores = np.asarray(scores.toarray() if hasattr(scores, "toarray") else scores).ravel()
        eligible = [i for i, r in enumerate(self._pool) if r.code3 == record.code3 and r.id != record.id]
        ranked = sorted(eligible, key=lambda i: (-scores[i], i))
        return [self._pool[i] for i in ranked[:k]]


class TfidfRetriever(_SimilarityRetriever):
    def __init__(self, pool: list[Record]):
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
        super().__init__(pool)

    def _embed_pool(self, texts):
        return self._vectorizer.fit_transform(texts)

    def _embed_query(self, text):
        return self._vectorizer.transform([text])


class SbertRetriever(_SimilarityRetriever):
    """Sentence-embedding retrieval. Install with `pip install sentence-transformers`."""

    def __init__(self, pool: list[Record], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "SbertRetriever needs the optional sentence-transformers package: pip install sentence-transformers"
            ) from error
        self._model = SentenceTransformer(model_name)
        super().__init__(pool)

    def _embed_pool(self, texts):
        return self._model.encode(texts, normalize_embeddings=True)

    def _embed_query(self, text):
        return self._model.encode([text], normalize_embeddings=True)
