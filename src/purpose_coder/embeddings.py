"""Sentence embeddings. An embedder maps a list of texts to an (n, d) float array.

Similarity selection and Vote-K both take an embedder as a plain function, so tests pass a fake one and
sentence-transformers stays an optional extra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Embed = Callable[[Sequence[str]], np.ndarray]

# The 2023 write-up says "the standard pre-trained sentence transformer from SBERT" without naming it.
# all-MiniLM-L6-v2 is the model in the sentence-transformers quickstart; the name here is
# not specified in the 2023 drafts.
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


def sbert_embedder(model_name: str = DEFAULT_SBERT_MODEL) -> Embed:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "SBERT embeddings need sentence-transformers: pip install 'oecd-aid-finance-llm-classifier[sbert]'"
        ) from error
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=float)

    return embed


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)
