"""Official OECD CRS purpose-code list, loaded from config/purpose_codes.json.

The config holds the whole list (all sectors), built by scripts/build_taxonomy.py from
https://webfs.oecd.org/oda/DataCollection/Resources/DAC-CRS-CODES.xlsx. Nothing here is hand-typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "purpose_codes.json"


@dataclass(frozen=True)
class PurposeCode:
    code: str
    name: str
    description: str
    parent: str
    voluntary: bool = False


class Taxonomy:
    def __init__(self, sectors: dict[str, str], codes: list[PurposeCode]):
        self._sectors = dict(sectors)
        self._codes = {c.code: c for c in codes}
        self._children: dict[str, list[PurposeCode]] = {}
        for code in codes:
            self._children.setdefault(code.parent, []).append(code)

    def children(self, code3: str) -> list[PurposeCode]:
        """Five-digit codes under a three-digit sector, in the OECD list's order (voluntary codes included)."""
        if code3 not in self._children:
            raise KeyError(f"no purpose codes under sector {code3!r}")
        return list(self._children[code3])

    def parent(self, code5: str) -> str:
        return self._codes[code5].parent

    def name(self, code: str) -> str:
        """Official OECD name of a 5-digit purpose code or a 3-digit sector."""
        if code in self._codes:
            return self._codes[code].name
        return self._sectors[code]

    def get(self, code5: str) -> PurposeCode:
        return self._codes[code5]


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    raw = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    codes = [PurposeCode(**c) for c in raw["codes"]]
    return Taxonomy(raw["sectors"], codes)
