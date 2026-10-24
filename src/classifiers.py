"""Classifiers behind one seam: fit on labelled records, predict a 5-digit code with a confidence.

The 3-digit sector of every record is known, so each adapter only chooses among the 5-digit children of that sector
that appear in the training data. A prediction outside the record's sector is impossible by construction.

- TfidfLogReg: one TF-IDF + logistic regression model per sector.
- TfidfKnn: one TF-IDF index per sector, cosine similarity, similarity-weighted vote of the k nearest records.
- FewShotLLM: builds the few-shot prompt, calls a TextGenerator, parses the 5-digit code. Output that is not a
  valid candidate becomes an abstention (code5=None), counted as wrong in accuracy and in `parse_failures`.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import Record
from src.llm import TextGenerator
from src.prompting import build_prompt
from src.retrieval import Retriever
from src.taxonomy import Taxonomy, load_taxonomy


@dataclass(frozen=True)
class Prediction:
    """code5 is None when the classifier abstains (no valid answer)."""

    code5: str | None
    confidence: float


class Classifier(Protocol):
    def fit(self, train: list[Record]) -> Classifier: ...

    def predict(self, records: list[Record]) -> list[Prediction]: ...


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2))


def _candidates_by_sector(train: list[Record], taxonomy: Taxonomy) -> dict[str, list[str]]:
    """Per sector: the sector's children (OECD order) that occur in the training data."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in train:
        seen[record.code3].add(record.code5)
    out = {}
    for code3, labels in seen.items():
        out[code3] = [c.code for c in taxonomy.children(code3) if c.code in labels]
    return out


class _PerSector:
    """Fits one sub-model per sector; subclasses define how a sector model is fitted and how it predicts."""

    def __init__(self, taxonomy: Taxonomy | None = None):
        self._taxonomy = taxonomy or load_taxonomy()
        self._models: dict[str, object] = {}

    def _fit_sector(self, texts: list[str], labels: list[str]):
        raise NotImplementedError

    def _predict_sector(self, model, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError

    def fit(self, train: list[Record]):
        candidates = _candidates_by_sector(train, self._taxonomy)
        by_sector: dict[str, list[Record]] = defaultdict(list)
        for record in train:
            if record.code5 in candidates.get(record.code3, ()):
                by_sector[record.code3].append(record)
        self._models = {
            code3: self._fit_sector([r.text for r in rs], [r.code5 for r in rs]) for code3, rs in by_sector.items()
        }
        return self

    def predict(self, records: list[Record]) -> list[Prediction]:
        out: list[Prediction | None] = [None] * len(records)
        positions: dict[str, list[int]] = defaultdict(list)
        for i, record in enumerate(records):
            positions[record.code3].append(i)
        for code3, idx in positions.items():
            if code3 not in self._models:
                for i in idx:
                    out[i] = Prediction(None, 0.0)
                continue
            for i, prediction in zip(idx, self._predict_sector(self._models[code3], [records[i].text for i in idx])):
                out[i] = prediction
        return out  # type: ignore[return-value]


class TfidfLogReg(_PerSector):
    """C=10 with balanced class weights; C was picked from {1, 10, 100} on val."""

    def __init__(self, taxonomy: Taxonomy | None = None, C: float = 10.0):
        super().__init__(taxonomy)
        self._C = C

    def _fit_sector(self, texts, labels):
        if len(set(labels)) == 1:
            return ("constant", labels[0])
        vectorizer = _vectorizer()
        matrix = vectorizer.fit_transform(texts)
        model = LogisticRegression(C=self._C, class_weight="balanced", max_iter=2000)
        model.fit(matrix, labels)
        return ("model", vectorizer, model)

    def _predict_sector(self, model, texts):
        if model[0] == "constant":
            return [Prediction(model[1], 1.0) for _ in texts]
        _, vectorizer, classifier = model
        probabilities = classifier.predict_proba(vectorizer.transform(texts))
        best = probabilities.argmax(axis=1)
        return [Prediction(str(classifier.classes_[b]), float(p[b])) for b, p in zip(best, probabilities)]


class TfidfKnn(_PerSector):
    """Cosine similarity over TF-IDF vectors; the k nearest training records vote, weighted by similarity.
    k=10 was picked from {3, 5, 10} on val.

    Confidence is the winning class's share of the total similarity weight among the k neighbours. With no word
    overlap at all, the sector's most common training code is returned with confidence 0.
    """

    def __init__(self, taxonomy: Taxonomy | None = None, k: int = 10):
        super().__init__(taxonomy)
        self._k = k

    def _fit_sector(self, texts, labels):
        vectorizer = _vectorizer()
        return (vectorizer, vectorizer.fit_transform(texts), labels)

    def _predict_sector(self, model, texts):
        vectorizer, matrix, labels = model
        majority = Counter(labels).most_common(1)[0][0]
        similarities = (vectorizer.transform(texts) @ matrix.T).toarray()
        out = []
        for row in similarities:
            nearest = np.argsort(-row, kind="stable")[: self._k]
            votes: dict[str, float] = defaultdict(float)
            for j in nearest:
                votes[labels[j]] += float(row[j])
            total = sum(votes.values())
            if total <= 0:
                out.append(Prediction(majority, 0.0))
                continue
            winner = max(sorted(votes), key=lambda c: votes[c])
            out.append(Prediction(winner, votes[winner] / total))
        return out


_CODE = re.compile(r"(?<!\d)\d{5}(?!\d)")


class FewShotLLM:
    """k most similar labelled examples as input/output pairs, then the project to code; the model answers a code.

    `retriever_factory(train)` builds the retriever over the training records when `fit` is called.
    """

    def __init__(
        self,
        generator: TextGenerator,
        retriever_factory: Callable[[list[Record]], Retriever],
        k: int = 5,
        taxonomy: Taxonomy | None = None,
    ):
        self._generator = generator
        self._retriever_factory = retriever_factory
        self._k = k
        self._taxonomy = taxonomy or load_taxonomy()
        self._retriever: Retriever | None = None
        self._candidates: dict[str, list[str]] = {}
        self.parse_failures = 0

    def fit(self, train: list[Record]) -> FewShotLLM:
        self._retriever = self._retriever_factory(train)
        self._candidates = _candidates_by_sector(train, self._taxonomy)
        self.parse_failures = 0
        return self

    def prompt_for(self, record: Record) -> str:
        assert self._retriever is not None, "call fit() first"
        valid = set(self._candidates.get(record.code3, ()))
        offered = [c for c in self._taxonomy.children(record.code3) if c.code in valid]
        examples = self._retriever.top_k(record, self._k)
        return build_prompt(record, examples, self._taxonomy, candidates=offered)

    def predict(self, records: list[Record]) -> list[Prediction]:
        askable = [i for i, r in enumerate(records) if self._candidates.get(r.code3)]
        answers = self._generator.generate([self.prompt_for(records[i]) for i in askable]) if askable else []
        out = [Prediction(None, 0.0)] * len(records)
        for i, answer in zip(askable, answers):
            match = _CODE.search(answer)
            if match and match.group() in self._candidates[records[i].code3]:
                out[i] = Prediction(match.group(), 1.0)
            else:
                self.parse_failures += 1
        return out
