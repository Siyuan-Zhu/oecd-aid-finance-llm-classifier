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
from sklearn.linear_model import LogisticRegression

from src.data import Record
from src.llm import TextGenerator
from src.prompting import Prompt, build_prompt
from src.retrieval import Retriever
from src.settings import make_vectorizer
from src.taxonomy import Taxonomy, load_taxonomy


@dataclass(frozen=True)
class Prediction:
    """code5 is None when the classifier abstains (no valid answer)."""

    code5: str | None
    confidence: float


class Classifier(Protocol):
    def fit(self, train: list[Record]) -> Classifier: ...

    def predict(self, records: list[Record]) -> list[Prediction]: ...


def _candidates_by_sector(train: list[Record], taxonomy: Taxonomy) -> dict[str, list[str]]:
    """Per sector: the sector's children (OECD order) that occur in the training data."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in train:
        seen[record.code3].add(record.code5)
    return {
        code3: [c.code for c in taxonomy.children(code3) if c.code in labels] for code3, labels in seen.items()
    }


class _SectorModel(Protocol):
    def predict(self, texts: list[str]) -> list[Prediction]: ...


class _PerSector:
    """Fits one sub-model per sector. Subclasses define `_fit_sector(texts, labels) -> _SectorModel`."""

    def __init__(self, taxonomy: Taxonomy | None = None):
        self._taxonomy = taxonomy or load_taxonomy()
        self._models: dict[str, _SectorModel] = {}

    def _fit_sector(self, texts: list[str], labels: list[str]) -> _SectorModel:
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
        out = [Prediction(None, 0.0)] * len(records)    # a sector never seen in training abstains
        positions: dict[str, list[int]] = defaultdict(list)
        for i, record in enumerate(records):
            positions[record.code3].append(i)
        for code3, idx in positions.items():
            if code3 in self._models:
                for i, prediction in zip(idx, self._models[code3].predict([records[i].text for i in idx])):
                    out[i] = prediction
        return out


class _ConstantSector:
    """A sector with one training code: always answers it."""

    def __init__(self, label: str):
        self._label = label

    def predict(self, texts):
        return [Prediction(self._label, 1.0) for _ in texts]


class _LogRegSector:
    def __init__(self, texts: list[str], labels: list[str], C: float):
        self._vectorizer = make_vectorizer()
        self._model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        self._model.fit(self._vectorizer.fit_transform(texts), labels)

    def predict(self, texts):
        probabilities = self._model.predict_proba(self._vectorizer.transform(texts))
        best = probabilities.argmax(axis=1)
        return [Prediction(str(self._model.classes_[b]), float(p[b])) for b, p in zip(best, probabilities)]


class TfidfLogReg(_PerSector):
    """TF-IDF features, balanced class weights, regularisation strength C (smaller is stronger).

    The default C is a starting point. The pipeline sweeps C on val and uses the best value.
    """

    def __init__(self, taxonomy: Taxonomy | None = None, C: float = 10.0):
        super().__init__(taxonomy)
        self._C = C

    def _fit_sector(self, texts, labels):
        if len(set(labels)) == 1:
            return _ConstantSector(labels[0])
        return _LogRegSector(texts, labels, self._C)


class _KnnSector:
    def __init__(self, texts: list[str], labels: list[str], k: int):
        self._vectorizer = make_vectorizer()
        self._matrix = self._vectorizer.fit_transform(texts)
        self._labels = labels
        self._k = k
        self._majority = Counter(labels).most_common(1)[0][0]

    def predict(self, texts):
        similarities = (self._vectorizer.transform(texts) @ self._matrix.T).toarray()
        out = []
        for row in similarities:
            votes: dict[str, float] = defaultdict(float)
            for j in np.argsort(-row, kind="stable")[: self._k]:
                votes[self._labels[j]] += float(row[j])
            total = sum(votes.values())
            if total <= 0:
                out.append(Prediction(self._majority, 0.0))
                continue
            winner = max(sorted(votes), key=lambda c: votes[c])
            out.append(Prediction(winner, votes[winner] / total))
        return out


class TfidfKnn(_PerSector):
    """Cosine similarity over TF-IDF vectors; the k nearest training records vote, weighted by similarity.

    Confidence is the winning class's share of the total similarity weight among the k neighbours. With no word
    overlap at all, the sector's most common training code is returned with confidence 0. The default k is a
    starting point. The pipeline sweeps k on val and uses the best value.
    """

    def __init__(self, taxonomy: Taxonomy | None = None, k: int = 10):
        super().__init__(taxonomy)
        self._k = k

    def _fit_sector(self, texts, labels):
        return _KnnSector(texts, labels, self._k)


_CODE = re.compile(r"(?<!\d)\d{5}(?!\d)")


class FewShotPrompter:
    """Builds the few-shot prompt for a record: sector criteria plus its k most similar labelled examples.

    `retriever_factory(train)` is called in `fit`, because a retriever needs the training pool to search. That is
    why this takes a factory and not a ready-made retriever.
    """

    def __init__(
        self,
        retriever_factory: Callable[[list[Record]], Retriever],
        k: int = 5,
        taxonomy: Taxonomy | None = None,
    ):
        self._retriever_factory = retriever_factory
        self._k = k
        self._taxonomy = taxonomy or load_taxonomy()
        self._retriever: Retriever | None = None
        self.candidates: dict[str, list[str]] = {}

    def fit(self, train: list[Record]) -> FewShotPrompter:
        self._retriever = self._retriever_factory(train)
        self.candidates = _candidates_by_sector(train, self._taxonomy)
        return self

    def prompt_for(self, record: Record) -> Prompt:
        if self._retriever is None:
            raise RuntimeError("call fit() first")
        valid = set(self.candidates.get(record.code3, ()))
        offered = [c for c in self._taxonomy.children(record.code3) if c.code in valid]
        return build_prompt(record, self._retriever.top_k(record, self._k), self._taxonomy, candidates=offered)


class FewShotLLM:
    """The prompt from FewShotPrompter, sent to a TextGenerator; the model answers with a 5-digit code.

    The constructor takes a retriever factory instead of a retriever because the retriever searches the training
    records, which exist only when `fit(train)` is called.
    """

    def __init__(
        self,
        generator: TextGenerator,
        retriever_factory: Callable[[list[Record]], Retriever],
        k: int = 5,
        taxonomy: Taxonomy | None = None,
    ):
        self._generator = generator
        self._prompter = FewShotPrompter(retriever_factory, k, taxonomy)
        self.parse_failures = 0

    def fit(self, train: list[Record]) -> FewShotLLM:
        self._prompter.fit(train)
        self.parse_failures = 0
        return self

    def predict(self, records: list[Record]) -> list[Prediction]:
        candidates = self._prompter.candidates
        askable = [i for i, r in enumerate(records) if candidates.get(r.code3)]
        prompts = [self._prompter.prompt_for(records[i]).text for i in askable]
        answers = self._generator.generate(prompts) if prompts else []
        out = [Prediction(None, 0.0)] * len(records)
        for i, answer in zip(askable, answers):
            match = _CODE.search(answer)
            if match and match.group() in candidates[records[i].code3]:
                out[i] = Prediction(match.group(), 1.0)
            else:
                self.parse_failures += 1
        return out
