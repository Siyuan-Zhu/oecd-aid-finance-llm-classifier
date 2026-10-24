"""build_sample filters the OECD bulk file down to the committed sample, and reports every count."""

import hashlib
import io
import zipfile

import pytest

from scripts.build_sample import build_sample, iter_crs_rows, looks_english, open_member
from src.data import ChecksumError, verify_sha256

ENGLISH = (
    "This project supports the national ministry of health to strengthen the delivery of services in the "
    "districts and to train the staff who work with the communities that are affected by the disease."
)
FRENCH = (
    "Ce projet soutient le ministère de la santé pour renforcer les services dans les districts et pour "
    "former le personnel qui travaille avec les communautés qui sont touchées par la maladie dans le pays."
)


def row(crs_id="1", sector="122", purpose="12262", title="Malaria project", text=ENGLISH, donor="Donor", year="2024"):
    return {
        "Year": year, "DonorName": donor, "CrsID": crs_id, "ProjectTitle": title,
        "SectorCode": sector, "PurposeCode": purpose, "LongDescription": text,
    }


def many(code, sector, n, prefix):
    """n rows with distinct English texts, so no filter except the minimum-per-code one touches them."""
    return [row(f"{prefix}{i}", sector, code, f"{prefix} title {i}", f"{ENGLISH} Case {prefix} number {chr(97 + i % 26)}{i}.")
            for i in range(n)]


def counts(report):
    return {name: n for name, n in report}


# ---- reading the bulk file ----

def test_iter_crs_rows_parses_quoted_pipe_delimited_lines_and_counts_malformed_ones():
    text = (
        '"Year"|"CrsID"|"SectorCode"\r\n'
        '"2024"|"A1"|"122"\r\n'
        '"2024"|"A2"\r\n'
        '"2024"|"A3"|"121"\r\n'
    )
    stats = {}
    rows = list(iter_crs_rows(io.StringIO(text), stats))
    assert [r["CrsID"] for r in rows] == ["A1", "A3"]
    assert stats == {"parsed": 2, "malformed": 1}


def test_open_member_reads_the_text_file_inside_a_zip(tmp_path):
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("CRS 2024 data.txt", '"a"|"b"\r\n"é"|"2"\r\n')
    with open_member(path, "CRS 2024 data.txt") as handle:
        assert handle.read() == '"a"|"b"\r\n"é"|"2"\r\n'


def test_verify_sha256_rejects_a_file_that_is_not_the_pinned_one(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    verify_sha256(path, hashlib.sha256(b"hello").hexdigest())
    with pytest.raises(ChecksumError):
        verify_sha256(path, "0" * 64)


# ---- the language heuristic ----

def test_looks_english_accepts_english_and_rejects_french():
    assert looks_english(ENGLISH)
    assert not looks_english(FRENCH)


# ---- filters, in order, with counts ----

def test_each_filter_removes_what_it_says_and_the_report_counts_it():
    rows = (
        many("12262", "122", 40, "A")
        + [row("B1", "110", "11110")]                                    # not a health sector
        + [row("B2", "122", "12110")]                                    # purpose is not a child of the sector
        + [row("B3", "122", "122")]                                      # stray aggregate label
        + [row("B4", text="Too short to keep.")]                         # under 20 words
        + [row("B5", text=FRENCH)]                                       # not English
        + [row("B6", title="TC AGGREGATED ACTIVITIES", text="TC AGGREGATED ACTIVITIES " + ENGLISH)]  # boilerplate
    )
    sample, report = build_sample(rows, min_per_code=30, cap=250, seed=1)
    c = counts(report)
    assert c["raw rows"] == 46
    assert c["health sectors 121/122/123"] == 45
    assert c["purpose code is a 5-digit child of the sector"] == 43
    assert c["long description has at least 20 words"] == 42
    assert c["English by stopword heuristic"] == 41
    assert c["boilerplate removed"] == 40
    assert len(sample) == 40


def test_identical_text_keeps_one_row_and_text_with_conflicting_codes_is_dropped():
    same = [row(f"S{i}", purpose="12262") for i in range(3)]               # same text, same code: keep one
    other = ENGLISH + " A different second project."
    conflict = [row("C1", purpose="12262", text=other), row("C2", purpose="12250", text=other)]
    filler = many("12262", "122", 35, "F")
    sample, report = build_sample(same + conflict + filler, min_per_code=30, cap=250, seed=1)
    c = counts(report)
    assert c["texts with conflicting purpose codes dropped"] == 38      # 3 + 35 survive, the 2 conflicting rows go
    assert c["one row per distinct text"] == 36
    assert sum(1 for r in sample if r["title"] == "Malaria project") == 1


def test_codes_below_the_minimum_are_dropped_and_listed():
    rows = many("12262", "122", 35, "M") + many("12263", "122", 5, "T")
    sample, report = build_sample(rows, min_per_code=30, cap=250, seed=1)
    assert {r["purpose_code"] for r in sample} == {"12262"}
    assert counts(report)["codes with at least 30 distinct texts"] == 35
    assert report.dropped_codes == {"12263": 5}


def test_cap_per_code_is_applied_with_a_fixed_seed():
    rows = many("12262", "122", 60, "M")
    first, _ = build_sample(rows, min_per_code=30, cap=40, seed=7)
    again, _ = build_sample(list(reversed(rows)), min_per_code=30, cap=40, seed=7)
    assert len(first) == 40
    assert first == again


def test_sample_rows_use_the_committed_column_names():
    sample, _ = build_sample(many("12262", "122", 31, "M"), min_per_code=30, cap=250, seed=1)
    assert list(sample[0]) == [
        "crs_id", "year", "donor_name", "sector_code", "purpose_code", "title", "long_description",
    ]
    assert sample[0]["sector_code"] == "122"
