"""The committed CRS health-sector sample and its train/val/test split.

`load_splits()` reads data/crs_health_2024_sample.csv, checks its SHA-256 against the pinned value, and splits it
60/20/20 into train, val and test. The split is stratified by 5-digit purpose code and grouped so that near-duplicate
texts stay together. CRS repeats text heavily, and a leaked duplicate inflates every score.

Grouping: two texts are joined when their TF-IDF cosine similarity (the vectoriser in src/settings.py, fitted on the
whole sample, over title plus description) is at least NEAR_DUPLICATE_COSINE. Groups are the connected components of
that relation, so a chain of similar texts forms one group. The guarantee is exact for that vectoriser: every pair of
texts in different splits has cosine below the threshold. A different vectoriser can score such a pair higher.

scripts/build_sample.py rebuilds the CSV from the official OECD bulk file.
"""

from __future__ import annotations

import csv
import hashlib
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from src.settings import make_vectorizer

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "crs_health_2024_sample.csv"
SAMPLE_SHA256 = "31a8a33651eeaf5b8be36b1be93530b6e55e7452d8bf3dec72b3f25f3c5eaf57"
SAMPLE_COLUMNS = ("crs_id", "year", "donor_name", "sector_code", "purpose_code", "title", "long_description")
SEED = 42
NEAR_DUPLICATE_COSINE = 0.8


class ChecksumError(RuntimeError):
    """The file on disk is not the file this code was written against."""


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    code3: str
    code5: str
    year: int
    donor: str


@dataclass(frozen=True)
class Splits:
    train: list[Record]
    val: list[Record]
    test: list[Record]


def normalise_text(text: str) -> str:
    return " ".join(text.split())


def near_duplicate_groups(texts: list[str], threshold: float = NEAR_DUPLICATE_COSINE) -> list[int]:
    """Group id per text: connected components of the 'TF-IDF cosine >= threshold' relation (union-find)."""
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    matrix = make_vectorizer().fit_transform(texts)
    similarity = (matrix @ matrix.T).tocoo()
    for i, j, value in zip(similarity.row, similarity.col, similarity.data):
        if i < j and value >= threshold:
            parent[find(i)] = find(j)
    ids: dict[int, int] = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(len(texts))]


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ChecksumError(f"{path} has SHA-256 {digest.hexdigest()}, expected {expected}")


def _read_records(path: Path) -> list[Record]:
    seen: Counter = Counter()
    records = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seen[row["crs_id"]] += 1
            title, description = normalise_text(row["title"]), normalise_text(row["long_description"])
            records.append(
                Record(
                    id=f"{row['crs_id']}-{seen[row['crs_id']]}",
                    text=f"{title}. {description}" if title else description,
                    code3=row["sector_code"],
                    code5=row["purpose_code"],
                    year=int(row["year"]),
                    donor=row["donor_name"],
                )
            )
    return records


def load_splits(path: Path | None = None, expected_sha256: str | None = SAMPLE_SHA256, seed: int = SEED) -> Splits:
    path = Path(path or SAMPLE_PATH)
    if expected_sha256 is not None:
        verify_sha256(path, expected_sha256)
    records = _read_records(path)
    labels = [r.code5 for r in records]
    groups = near_duplicate_groups([r.text for r in records])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    parts: list[list[Record]] = [[] for _ in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # a class with fewer than 5 groups cannot be spread over 5 folds
        for fold, (_, held_out) in enumerate(folds.split(records, labels, groups)):
            parts[fold] = [records[i] for i in held_out]
    test, val = parts[0], parts[1]
    train = [r for part in parts[2:] for r in part]
    return Splits(train=train, val=val, test=test)


def split_statistics(splits: Splits) -> dict:
    """Group sizes, and the highest TF-IDF cosine between any val or test text and any train text."""
    texts = [r.text for r in splits.train + splits.val + splits.test]
    sizes = Counter(near_duplicate_groups(texts))
    matrix = make_vectorizer().fit_transform(texts)
    n_train, n_val = len(splits.train), len(splits.val)
    train = matrix[:n_train]
    worst = (matrix[n_train:] @ train.T).max(axis=1).toarray().ravel()
    return {
        "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
        "components": len(sizes),
        "largest_component": max(sizes.values()),
        "singleton_components": sum(1 for size in sizes.values() if size == 1),
        "max_cosine_to_train": {
            "val": round(float(worst[:n_val].max()), 4),
            "test": round(float(worst[n_val:].max()), 4),
        },
    }
