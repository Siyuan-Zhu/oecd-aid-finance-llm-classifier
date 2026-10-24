"""The committed CRS health-sector sample and its train/val/test split.

`load_splits()` reads data/crs_health_2024_sample.csv, checks its SHA-256 against the pinned value, and splits it
60/20/20 into train, val and test. The split is stratified by 5-digit purpose code and grouped by a normalised
description key, so near-identical descriptions (the same programme reported for several years or donors) never
land in two different splits. CRS repeats text heavily, and a leaked duplicate inflates every score.

scripts/build_sample.py rebuilds the CSV from the official OECD bulk file.
"""

from __future__ import annotations

import csv
import hashlib
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "crs_health_2024_sample.csv"
SAMPLE_SHA256 = "31a8a33651eeaf5b8be36b1be93530b6e55e7452d8bf3dec72b3f25f3c5eaf57"
SAMPLE_COLUMNS = ("crs_id", "year", "donor_name", "sector_code", "purpose_code", "title", "long_description")
SEED = 42
GROUP_KEY_CHARS = 200


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


def group_key(text: str) -> str:
    """Description lower-cased with digits and punctuation removed, first 200 characters."""
    stripped = re.sub(r"[\d_]+|[^\w\s]", "", text.lower())
    return normalise_text(stripped)[:GROUP_KEY_CHARS]


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
    groups = [group_key(r.text) for r in records]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    parts: list[list[Record]] = [[] for _ in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # a class with fewer than 5 groups cannot be spread over 5 folds
        for fold, (_, held_out) in enumerate(folds.split(records, labels, groups)):
            parts[fold] = [records[i] for i in held_out]
    test, val = parts[0], parts[1]
    train = [r for part in parts[2:] for r in part]
    return Splits(train=train, val=val, test=test)
