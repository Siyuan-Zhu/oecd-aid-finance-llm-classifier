"""Classifiers behind one seam: fit on labelled records, predict a 5-digit code with a confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.data import Record


@dataclass(frozen=True)
class Prediction:
    """code5 is None when the classifier abstains (no valid answer)."""

    code5: str | None
    confidence: float


class Classifier(Protocol):
    def fit(self, train: list[Record]) -> Classifier: ...

    def predict(self, records: list[Record]) -> list[Prediction]: ...
