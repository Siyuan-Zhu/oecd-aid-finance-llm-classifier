"""data.load_splits: checksum, record shape, and splits that never share a description across train/val/test."""

import csv
import hashlib
import itertools
from collections import Counter

import pytest

from src.data import SAMPLE_COLUMNS, ChecksumError, Record, group_key, load_splits, normalise_text

LETTERS = "abcdefghij"
WORDS = ["".join(p) for p in itertools.product(LETTERS, repeat=3)]


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rows(per_code=50, codes=(("122", "12262"), ("122", "12263"), ("123", "12340"))):
    rows, n = [], 0
    for sector, code in codes:
        for _ in range(per_code):
            word = WORDS[n]
            rows.append({
                "crs_id": str(1000 + n), "year": "2024", "donor_name": "Donor",
                "sector_code": sector, "purpose_code": code,
                "title": f"Title {word}", "long_description": f"Description of the {word} programme.",
            })
            n += 1
    return rows


@pytest.fixture
def small_csv(tmp_path):
    path = tmp_path / "sample.csv"
    sha = write_csv(path, make_rows())
    return path, sha


def test_wrong_checksum_stops_the_load(small_csv):
    path, _ = small_csv
    with pytest.raises(ChecksumError):
        load_splits(path, expected_sha256="0" * 64)


def test_records_carry_text_codes_year_and_donor(small_csv):
    path, sha = small_csv
    splits = load_splits(path, expected_sha256=sha)
    record = next(r for r in splits.train + splits.val + splits.test if r.text.startswith("Title aaa."))
    assert record == Record(
        id=record.id, text="Title aaa. Description of the aaa programme.",
        code3="122", code5="12262", year=2024, donor="Donor",
    )


def test_splits_are_60_20_20_within_every_code(small_csv):
    path, sha = small_csv
    splits = load_splits(path, expected_sha256=sha)
    for code in ("12262", "12263", "12340"):
        sizes = [sum(r.code5 == code for r in part) for part in (splits.train, splits.val, splits.test)]
        assert sizes == [30, 10, 10]


def test_ids_are_unique_and_no_record_is_in_two_splits(small_csv):
    path, sha = small_csv
    splits = load_splits(path, expected_sha256=sha)
    ids = [r.id for part in (splits.train, splits.val, splits.test) for r in part]
    assert len(ids) == len(set(ids)) == 150


def test_same_crs_id_still_gets_distinct_record_ids(tmp_path):
    rows = make_rows(per_code=10, codes=(("122", "12262"),))
    rows[1]["crs_id"] = rows[0]["crs_id"]
    path = tmp_path / "s.csv"
    sha = write_csv(path, rows)
    splits = load_splits(path, expected_sha256=sha)
    ids = [r.id for part in (splits.train, splits.val, splits.test) for r in part]
    assert len(set(ids)) == 10


def test_descriptions_that_differ_only_in_digits_and_punctuation_share_a_split(tmp_path):
    rows = make_rows(per_code=60, codes=(("122", "12262"),))
    for i in range(12):
        rows.append({
            "crs_id": str(9000 + i), "year": "2024", "donor_name": "Donor", "sector_code": "122",
            "purpose_code": "12262", "title": f"Direct Aid Program (DAP) 20{i}-2{i}",
            "long_description": f"Small grants programme number {i}, funded in 20{i}!",
        })
    path = tmp_path / "s.csv"
    sha = write_csv(path, rows)
    splits = load_splits(path, expected_sha256=sha)
    homes = {name for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test))
             for r in part if r.text.startswith("Direct Aid Program")}
    assert len(homes) == 1


def test_splits_are_deterministic_and_the_seed_changes_them(small_csv):
    path, sha = small_csv
    a = load_splits(path, expected_sha256=sha)
    b = load_splits(path, expected_sha256=sha)
    c = load_splits(path, expected_sha256=sha, seed=1)
    assert a == b
    assert [r.id for r in a.test] != [r.id for r in c.test]


def test_group_key_ignores_case_digits_and_punctuation_and_keeps_200_characters():
    assert group_key("Direct Aid Program (DAP) 2023-24") == group_key("direct aid program DAP 2024-25")
    assert len(group_key("a" * 500)) == 200


def test_normalise_text_collapses_whitespace():
    assert normalise_text("  a \n\t b  ") == "a b"


# ---- the committed sample ----

def test_committed_sample_matches_its_pinned_checksum_and_has_no_group_in_two_splits():
    splits = load_splits()
    parts = {"train": splits.train, "val": splits.val, "test": splits.test}
    home = {}
    for name, part in parts.items():
        for r in part:
            home.setdefault(group_key(r.text), set()).add(name)
    assert all(len(v) == 1 for v in home.values())
    sizes = Counter({k: len(v) for k, v in parts.items()})
    total = sum(sizes.values())
    assert 0.55 < sizes["train"] / total < 0.65
    assert 0.15 < sizes["val"] / total < 0.25
    assert 0.15 < sizes["test"] / total < 0.25
