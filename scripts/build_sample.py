"""Rebuild data/crs_health_2024_sample.csv from the official OECD CRS 2024 bulk file.

Source: https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS%202024%20data.zip (46.5 MB, 2024 flows, dataflow
version v20261006). The host sits behind Cloudflare, which blocks curl. Python's urllib with a plain, non-browser
User-Agent and an `Upgrade-Insecure-Requests: 1` header gets through.

    python scripts/build_sample.py --zip path/to/CRS_2024_data.zip     # use a local copy
    python scripts/build_sample.py --download                          # fetch into data/raw/ first

Every filter prints the number of rows left after it. The same counts are recorded in DATA.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import SAMPLE_COLUMNS, normalise_text, verify_sha256  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402

ZIP_URL = "https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS%202024%20data.zip"
ZIP_SHA256 = "9e1387b4f3981854f9de786758e482add7a802e207f94d1bddaad9cef51fb9b5"
MEMBER = "CRS 2024 data.txt"
USER_AGENT = "oecd-aid-finance-llm-classifier/0.1 (research sample builder)"
OUT_PATH = REPO_ROOT / "data" / "crs_health_2024_sample.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"

HEALTH_SECTORS = ("121", "122", "123")
MIN_WORDS = 20
MIN_PER_CODE = 30
CAP_PER_CODE = 250
SEED = 42
NEEDED = ("Year", "DonorName", "CrsID", "ProjectTitle", "SectorCode", "PurposeCode", "LongDescription")

# Entries that name no project. Matched against the whole title or the whole description.
BOILERPLATE = re.compile(r"^\s*tc aggregated activities\s*$", re.IGNORECASE)

# Rough language test: count common function words of each language. English wins when it has at least as many
# hits as French and Spanish and the English hits are at least 10% of all words.
_EN = set(
    "the and of to in for with by is are on as be an will this from that which at or it has have their these also "
    "through its was were can been such between more other all not who into than each".split()
)
_FR = set("le la les des du de et en pour une un dans au aux est sur par avec qui que ce cette ces sont son ses leur leurs afin ainsi entre plus ou à été".split())
_ES = set("el los las del de y en para una un con por que se su sus es al como más entre este esta estos estas o".split())
_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def looks_english(text: str) -> bool:
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 5:
        return False
    en = sum(w in _EN for w in words)
    fr = sum(w in _FR for w in words)
    es = sum(w in _ES for w in words)
    return en > 0 and en >= max(fr, es) and en / len(words) >= 0.10


def download_zip(dest: Path, url: str = ZIP_URL) -> None:
    headers = {"User-Agent": USER_AGENT, "Upgrade-Insecure-Requests": "1"}
    request = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300) as response, open(dest, "wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)


@contextmanager
def open_member(zip_path: Path, member: str = MEMBER):
    """The bulk file as a UTF-8 text stream, read straight from the zip."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as raw:
        yield io.TextIOWrapper(raw, encoding="utf-8", newline="")


def iter_crs_rows(handle, stats: dict):
    """Rows of the pipe-delimited, fully quoted bulk file as dicts. Rows with the wrong field count are skipped."""
    csv.field_size_limit(10**9)
    reader = csv.reader(handle, delimiter="|", quotechar='"')
    header = next(reader)
    index = {name: i for i, name in enumerate(header)}
    stats.update(parsed=0, malformed=0)
    for row in reader:
        if len(row) != len(header):
            stats["malformed"] += 1
            continue
        stats["parsed"] += 1
        yield {name: row[index[name]] for name in NEEDED if name in index}


class Report:
    """Row counts after each filter, in order, plus the codes dropped for having too few examples."""

    def __init__(self):
        self.steps: list[tuple[str, int]] = []
        self.dropped_codes: dict[str, int] = {}
        self.available_per_code: dict[str, int] = {}
        self.final_per_code: dict[str, int] = {}

    def add(self, name: str, n: int) -> None:
        self.steps.append((name, n))

    def __iter__(self):
        return iter(self.steps)


def build_sample(rows, min_per_code: int = MIN_PER_CODE, cap: int = CAP_PER_CODE, seed: int = SEED):
    """Apply the sample rules. Returns (sample rows as dicts with the committed columns, Report)."""
    taxonomy = load_taxonomy()
    report = Report()

    raw = 0
    health = []
    for row in rows:
        raw += 1
        if row["SectorCode"].strip() in HEALTH_SECTORS:
            health.append(row)
    report.add("raw rows", raw)
    report.add("health sectors 121/122/123", len(health))

    def keep(name, rows_, predicate):
        kept = [r for r in rows_ if predicate(r)]
        report.add(name, len(kept))
        return kept

    def is_child(r):
        sector, purpose = r["SectorCode"].strip(), r["PurposeCode"].strip()
        return any(c.code == purpose for c in taxonomy.children(sector))

    rows_ = keep("purpose code is a 5-digit child of the sector", health, is_child)
    rows_ = keep("long description has at least 20 words", rows_, lambda r: word_count(r["LongDescription"]) >= MIN_WORDS)
    rows_ = keep("English by stopword heuristic", rows_, lambda r: looks_english(r["LongDescription"]))
    rows_ = keep(
        "boilerplate removed",
        rows_,
        lambda r: not (BOILERPLATE.match(r["ProjectTitle"]) or BOILERPLATE.match(r["LongDescription"])),
    )

    def text_of(r):
        title, description = normalise_text(r["ProjectTitle"]), normalise_text(r["LongDescription"])
        return f"{title}. {description}" if title else description

    codes_by_text = defaultdict(set)
    for r in rows_:
        codes_by_text[text_of(r).lower()].add(r["PurposeCode"].strip())
    conflicting = {t for t, codes in codes_by_text.items() if len(codes) > 1}
    rows_ = keep("texts with conflicting purpose codes dropped", rows_, lambda r: text_of(r).lower() not in conflicting)

    first_by_text = {}
    for r in rows_:
        first_by_text.setdefault(text_of(r).lower(), r)
    rows_ = list(first_by_text.values())
    report.add("one row per distinct text", len(rows_))

    per_code = Counter(r["PurposeCode"].strip() for r in rows_)
    report.dropped_codes = {c: n for c, n in sorted(per_code.items()) if n < min_per_code}
    rows_ = keep(
        f"codes with at least {min_per_code} distinct texts",
        rows_,
        lambda r: per_code[r["PurposeCode"].strip()] >= min_per_code,
    )

    by_code = defaultdict(list)
    for r in rows_:
        by_code[r["PurposeCode"].strip()].append(r)
    chosen = []
    for code in sorted(by_code):
        pool = sorted(by_code[code], key=lambda r: (text_of(r).lower(), r["CrsID"]))
        if len(pool) > cap:
            pool = random.Random(f"{seed}:{code}").sample(pool, cap)
        chosen.extend(sorted(pool, key=lambda r: (text_of(r).lower(), r["CrsID"])))
    report.add(f"capped at {cap} per code", len(chosen))
    report.available_per_code = {code: len(rs) for code, rs in sorted(by_code.items())}
    report.final_per_code = {code: min(len(rs), cap) for code, rs in sorted(by_code.items())}

    sample = [
        {
            "crs_id": r["CrsID"].strip(),
            "year": r["Year"].strip(),
            "donor_name": r["DonorName"].strip(),
            "sector_code": r["SectorCode"].strip(),
            "purpose_code": r["PurposeCode"].strip(),
            "title": normalise_text(r["ProjectTitle"]),
            "long_description": normalise_text(r["LongDescription"]),
        }
        for r in chosen
    ]
    return sample, report


def write_sample(sample: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, dest="zip_path", help="local copy of CRS 2024 data.zip")
    parser.add_argument("--download", action="store_true", help=f"download the zip into {RAW_DIR.name}/ first")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    zip_path = args.zip_path
    if args.download:
        zip_path = RAW_DIR / "CRS_2024_data.zip"
        print(f"downloading {ZIP_URL}")
        download_zip(zip_path)
    if zip_path is None:
        parser.error("give --zip PATH or --download")
    verify_sha256(zip_path, ZIP_SHA256)
    stats: dict = {}
    with open_member(zip_path) as handle:
        sample, report = build_sample(iter_crs_rows(handle, stats))
    print(f"bulk file rows parsed: {stats['parsed']}, skipped as malformed (wrong field count): {stats['malformed']}")
    for name, n in report:
        print(f"{n:>9,}  {name}")
    print("codes dropped for too few distinct texts:", report.dropped_codes or "none")
    print("distinct texts per code before the cap:", report.available_per_code)
    print("final rows per code:", report.final_per_code)
    digest = write_sample(sample, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes, {len(sample)} rows)")
    print(f"SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
