"""The two record types and the CSV readers.

Unlabelled CSV columns: title, description, code3 (the project's 3-digit sector code).
Example-pool CSV columns: title, description, code5, and an optional id. Without an id column the
1-based row number is the id. Extra columns are allowed and ignored.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def join_text(title: str, description: str) -> str:
    """Title and description as one input, joined the way the 2023 prompts joined them."""
    return f"{title.strip()} ; {description.strip()}"


@dataclass(frozen=True)
class Project:
    title: str
    description: str
    code3: str

    @property
    def text(self) -> str:
        return join_text(self.title, self.description)


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    description: str
    code5: str

    @property
    def code3(self) -> str:
        return self.code5[:3]

    @property
    def text(self) -> str:
        return join_text(self.title, self.description)


def _read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        return [{k: (v or "").strip() for k, v in row.items() if k is not None} for row in reader]


def read_unlabelled(path: Path) -> tuple[list[dict[str, str]], list[Project]]:
    """The raw rows (kept so the output can carry every input column) and the projects they hold."""
    rows = _read_rows(path, ("title", "description", "code3"))
    return rows, [Project(r["title"], r["description"], r["code3"]) for r in rows]


def read_pool(path: Path) -> list[Example]:
    rows = _read_rows(path, ("title", "description", "code5"))
    examples = []
    for number, row in enumerate(rows, start=1):
        code5 = row["code5"]
        if not (code5.isdigit() and len(code5) == 5):
            raise ValueError(f"{path}: row {number} has code5 {code5!r}, expected 5 digits")
        examples.append(Example(row.get("id") or str(number), row["title"], row["description"], code5))
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: example ids are not unique")
    return examples
