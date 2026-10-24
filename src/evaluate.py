"""Scoring: accuracy, macro-F1, per-sector and per-code accuracy, confusion pairs, and the abstention threshold.

An abstention (code5 is None) counts as wrong in accuracy and macro-F1 and is reported separately. The abstention
threshold for the baselines is chosen on val and applied once to test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.classifiers import Classifier, Prediction
from src.data import Record


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    abstentions: int
    per_sector: dict[str, float] = field(default_factory=dict)
    per_code: dict[str, float] = field(default_factory=dict)
    confusions: list[tuple[str, str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "abstentions": self.abstentions,
            "per_sector_accuracy": {k: round(v, 4) for k, v in sorted(self.per_sector.items())},
            "per_code_accuracy": {k: round(v, 4) for k, v in sorted(self.per_code.items())},
            "top_confusions": [{"true": t, "predicted": p, "count": c} for t, p, c in self.confusions],
        }


@dataclass(frozen=True)
class Selective:
    """Result of answering only when confident: share of records answered, and accuracy among those answered."""

    answered: int
    coverage: float
    precision: float


def _check(predictions: list[Prediction], records: list[Record]) -> None:
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} predictions for {len(records)} records")


def evaluate(predictions: list[Prediction], records: list[Record], top_confusions: int = 10) -> Metrics:
    _check(predictions, records)
    n = len(records)
    truth = [r.code5 for r in records]
    guess = [p.code5 for p in predictions]
    hits = sum(t == g for t, g in zip(truth, guess))

    per_code_hits, per_code_n = Counter(), Counter()
    per_sector_hits, per_sector_n = Counter(), Counter()
    for record, g in zip(records, guess):
        ok = g == record.code5
        per_code_n[record.code5] += 1
        per_sector_n[record.code3] += 1
        per_code_hits[record.code5] += ok
        per_sector_hits[record.code3] += ok

    f1s = []
    for code in sorted(per_code_n):
        tp = per_code_hits[code]
        fp = sum(1 for t, g in zip(truth, guess) if g == code and t != code)
        fn = per_code_n[code] - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)

    confusion = defaultdict(int)
    for t, g in zip(truth, guess):
        if g is not None and g != t:
            confusion[(t, g)] += 1
    pairs = sorted(((t, g, c) for (t, g), c in confusion.items()), key=lambda x: (-x[2], x[0], x[1]))

    return Metrics(
        n=n,
        accuracy=hits / n if n else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        abstentions=sum(g is None for g in guess),
        per_sector={s: per_sector_hits[s] / per_sector_n[s] for s in per_sector_n},
        per_code={c: per_code_hits[c] / per_code_n[c] for c in per_code_n},
        confusions=pairs[:top_confusions],
    )


def apply_threshold(predictions: list[Prediction], threshold: float) -> list[Prediction]:
    """Abstain wherever confidence is below the threshold."""
    return [p if p.confidence >= threshold else Prediction(None, p.confidence) for p in predictions]


def selective(predictions: list[Prediction], records: list[Record]) -> Selective:
    _check(predictions, records)
    answered = [(p, r) for p, r in zip(predictions, records) if p.code5 is not None]
    correct = sum(p.code5 == r.code5 for p, r in answered)
    return Selective(
        answered=len(answered),
        coverage=len(answered) / len(records) if records else 0.0,
        precision=correct / len(answered) if answered else 0.0,
    )


def choose_threshold(predictions: list[Prediction], records: list[Record], target_precision: float = 0.9) -> float:
    """Lowest confidence threshold (widest coverage) whose precision on these records reaches the target.

    Candidates are the observed confidences. If none reaches the target, returns 0.0 (answer everything).
    Call this on val only.
    """
    _check(predictions, records)
    best_threshold, best_coverage = 0.0, -1.0
    for candidate in sorted({p.confidence for p in predictions if p.code5 is not None}):
        result = selective(apply_threshold(predictions, candidate), records)
        if result.answered and result.precision >= target_precision and result.coverage > best_coverage:
            best_threshold, best_coverage = candidate, result.coverage
    return best_threshold


@dataclass(frozen=True)
class Candidate:
    value: object
    val_accuracy: float
    val_macro_f1: float


@dataclass(frozen=True)
class Selection:
    metric: str
    best: object
    candidates: list[Candidate]


def select_by_validation(
    make: Callable[[object], Classifier],
    grid: Sequence,
    train: list[Record],
    val: list[Record],
    metric: str = "macro_f1",
) -> Selection:
    """Fit `make(value)` on train for every value in the grid and score it on val.

    The best value is the one with the highest val macro-F1 (or accuracy, if `metric="accuracy"`); ties go to the
    earlier value in the grid. Test records never enter here. The caller refits the winner on train and scores it
    on test once.
    """
    candidates = []
    for value in grid:
        scores = evaluate(make(value).fit(train).predict(val), val)
        candidates.append(Candidate(value, scores.accuracy, scores.macro_f1))
    key = (lambda c: c.val_macro_f1) if metric == "macro_f1" else (lambda c: c.val_accuracy)
    best = max(candidates, key=key)   # max() keeps the first of equal candidates
    return Selection(metric=metric, best=best.value, candidates=candidates)
