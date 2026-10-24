"""Rebuild config/purpose_codes.json from the official OECD DAC/CRS code list.

Source: https://webfs.oecd.org/oda/DataCollection/Resources/DAC-CRS-CODES.xlsx, sheet "Purpose codes".
Needs openpyxl (a script-only dependency, not in requirements.txt).

    python scripts/build_taxonomy.py --xlsx path/to/DAC-CRS-CODES.xlsx
    python scripts/build_taxonomy.py --download        # fetch the workbook first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "config" / "purpose_codes.json"

SOURCE_URL = "https://webfs.oecd.org/oda/DataCollection/Resources/DAC-CRS-CODES.xlsx"
SOURCE_SHA256 = "eb5d320eeebb93f97f6b4a8e0ff4957cf923a4a59efa35f7bb3fab7d7c5f7b52"
RETRIEVED = "2026-10-23"
SHEET = "Purpose codes"


def _clean(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _digits(value, length: int) -> str | None:
    text = _clean(value)
    return text if text.isdigit() and len(text) == length else None


def parse_purpose_sheet(rows) -> dict:
    """Rows are (sector, core 5-digit code, voluntary 5-digit code, name, notes) tuples from the sheet."""
    sectors: dict[str, str] = {}
    codes: list[dict] = []
    parent = None
    for row in rows:
        row = tuple(row) + (None,) * (5 - len(row))
        sector = _digits(row[0], 3)
        if sector:
            parent = sector
            sectors[sector] = _clean(row[3])
            continue
        core, voluntary = _digits(row[1], 5), _digits(row[2], 5)
        code = core or voluntary
        if code is None or parent is None:
            continue
        codes.append(
            {
                "code": code,
                "name": _clean(row[3]),
                "description": _clean(row[4]),
                "parent": parent,
                "voluntary": core is None,
            }
        )
    return {"sectors": sectors, "codes": codes}


def read_workbook(path: Path) -> dict:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[SHEET]
    rows = [r[:5] for r in sheet.iter_rows(values_only=True)]
    header_note = _clean(rows[1][0]) if len(rows) > 1 else ""
    parsed = parse_purpose_sheet(rows)
    parsed["source"] = {
        "url": SOURCE_URL,
        "sheet": SHEET,
        "sheet_note": header_note,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "retrieved": RETRIEVED,
    }
    return parsed


def download(dest: Path) -> None:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "oecd-aid-finance-llm-classifier/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=Path, help="local copy of DAC-CRS-CODES.xlsx")
    parser.add_argument("--download", action="store_true", help="download the workbook into data/raw/ first")
    args = parser.parse_args(argv)
    path = args.xlsx
    if args.download:
        path = REPO_ROOT / "data" / "raw" / "DAC-CRS-CODES.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        download(path)
    if path is None:
        parser.error("give --xlsx PATH or --download")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        print(f"warning: workbook SHA-256 {digest} differs from the pinned {SOURCE_SHA256}", file=sys.stderr)
    parsed = read_workbook(path)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(parsed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}: {len(parsed['sectors'])} sectors, {len(parsed['codes'])} purpose codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
