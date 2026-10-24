"""How the k in-context examples are chosen from the labelled pool.

A selector has one method, `select(project, pool, k) -> list[Example]`, and returns examples in the order they
should appear in the prompt. The 2023 work compared:

    zero-shot            k = 0, no selection
    hand-picked          examples chosen by hand after looking at errors (HandPicked)
    similarity-based     the k nearest pool projects by SBERT cosine similarity, as in Liu et al. 2021 (Similarity)
    reversed order       the same examples in the opposite order, to check recency bias (ReverseOrder)

Random selection (Random) is how the 2023 hand-picked run started: five random examples.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from purpose_coder.data import Example, Project
from purpose_coder.embeddings import Embed, normalise


class Selector(Protocol):
    def select(self, project: Project, pool: Sequence[Example], k: int) -> list[Example]: ...


class Random:
    """k pool examples drawn at random. One generator per selector, so a seed fixes the whole run."""

    def __init__(self, seed: int = 0):
        # The seed is not specified in the 2023 drafts.
        self._rng = random.Random(seed)

    def select(self, project: Project, pool: Sequence[Example], k: int) -> list[Example]:
        return self._rng.sample(list(pool), min(k, len(pool)))


class HandPicked:
    """The examples with the given ids, in the order given, limited to k. Ids missing from `pool` are skipped."""

    def __init__(self, ids: Sequence[str]):
        self.ids = list(ids)

    def select(self, project: Project, pool: Sequence[Example], k: int) -> list[Example]:
        by_id = {e.id: e for e in pool}
        return [by_id[i] for i in self.ids if i in by_id][:k]


class Similarity:
    """The k pool examples closest to the project by cosine similarity of their embeddings, closest first."""

    def __init__(self, embed: Embed):
        self._embed = embed
        self._cache: dict[str, np.ndarray] = {}

    def _pool_vectors(self, pool: Sequence[Example]) -> np.ndarray:
        missing = [e for e in pool if e.id not in self._cache]
        if missing:
            for example, vector in zip(missing, normalise(np.asarray(self._embed([e.text for e in missing])))):
                self._cache[example.id] = vector
        return np.stack([self._cache[e.id] for e in pool])

    def select(self, project: Project, pool: Sequence[Example], k: int) -> list[Example]:
        if k <= 0 or not pool:
            return []
        query = normalise(np.asarray(self._embed([project.text])))[0]
        scores = self._pool_vectors(pool) @ query
        # Stable sort on negated scores: ties keep pool order.
        order = np.argsort(-scores, kind="stable")[:k]
        return [pool[i] for i in order]


class ReverseOrder:
    """Wraps a selector and reverses its examples. With Similarity this puts the closest example last,
    next to the project being coded."""

    def __init__(self, inner: Selector):
        self.inner = inner

    def select(self, project: Project, pool: Sequence[Example], k: int) -> list[Example]:
        return list(reversed(self.inner.select(project, pool, k)))
