"""data.load_splits: checksum, record shape, and splits that never share a description across train/val/test."""

import csv
import hashlib
import itertools

import pytest

from src.data import (
    NEAR_DUPLICATE_COSINE, SAMPLE_COLUMNS, ChecksumError, Record, load_splits, near_duplicate_groups, normalise_text,
)
from src.settings import make_vectorizer

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


def test_descriptions_that_differ_only_in_the_title_share_a_split(tmp_path):
    rows = make_rows(per_code=60, codes=(("122", "12262"),))
    countries = ["Angola", "Mozambique", "Malawi", "Zambia", "Kenya", "Uganda", "Ghana", "Mali", "Niger", "Chad"]
    for i, country in enumerate(countries):
        rows.append({
            "crs_id": str(9000 + i), "year": "2024", "donor_name": "Donor", "sector_code": "122",
            "purpose_code": "12262", "title": f"Health Strengthening Program (PARES) - {country}",
            "long_description": "The programme trains community health workers, equips district clinics and "
                                "supports the national malaria response through supervision and supplies.",
        })
    path = tmp_path / "s.csv"
    sha = write_csv(path, rows)
    splits = load_splits(path, expected_sha256=sha)
    homes = {name for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test))
             for r in part if r.text.startswith("Health Strengthening Program")}
    assert len(homes) == 1


def test_splits_are_deterministic_and_the_seed_changes_them(small_csv):
    path, sha = small_csv
    a = load_splits(path, expected_sha256=sha)
    b = load_splits(path, expected_sha256=sha)
    c = load_splits(path, expected_sha256=sha, seed=1)
    assert a == b
    assert [r.id for r in a.test] != [r.id for r in c.test]


def test_near_duplicate_groups_join_texts_at_or_above_the_cosine_threshold_and_chain_them():
    texts = [
        "malaria bed nets distribution in rural villages with insecticide spraying",
        "malaria bed nets distribution in rural villages with insecticide spraying programme",
        "tuberculosis drug treatment clinics laboratory detection programme for patients",
        "tuberculosis drug treatment clinics laboratory detection programme for patients nationwide",
        "nutrition supplements for children under five",
    ]
    groups = near_duplicate_groups(texts)
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert len({groups[0], groups[2], groups[4]}) == 3


def test_normalise_text_collapses_whitespace():
    assert normalise_text("  a \n\t b  ") == "a b"


# ---- the committed sample ----

def test_committed_sample_matches_its_pinned_checksum_and_splits_roughly_60_20_20():
    splits = load_splits()
    sizes = {name: len(part) for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test))}
    total = sum(sizes.values())
    assert 0.55 < sizes["train"] / total < 0.65
    assert 0.15 < sizes["val"] / total < 0.25
    assert 0.15 < sizes["test"] / total < 0.25


def test_no_val_or_test_text_has_a_train_neighbour_at_or_above_the_cosine_threshold():
    splits = load_splits()
    matrix = make_vectorizer().fit_transform([r.text for r in splits.train + splits.val + splits.test])
    n_train = len(splits.train)
    train, held_out = matrix[:n_train], matrix[n_train:]
    worst = (held_out @ train.T).max(axis=1).toarray().ravel()
    assert worst.max() < NEAR_DUPLICATE_COSINE
