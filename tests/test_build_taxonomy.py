"""build_taxonomy turns the OECD purpose-code sheet into the committed config."""

from scripts.build_taxonomy import parse_purpose_sheet

SHEET = [
    ("LIST OF CRS PURPOSE CODES", None, None, None, None),
    ("DAC 5", "CRS", "voluntary code", "DESCRIPTION", "Clarifications"),
    (120, None, None, "Health", None),
    (121, None, None, "Health, General", None),
    (None, "12110", None, "Health policy and administrative management ", "Health sector policy.\n"),
    (None, None, "12196", "Health statistics and data", "Statistics."),
    (122, None, None, "Basic Health", None),
    (None, 12220, None, "Basic health care", None),
]


def test_five_digit_codes_get_the_sector_they_sit_under():
    parsed = parse_purpose_sheet(SHEET)
    by_code = {c["code"]: c for c in parsed["codes"]}
    assert by_code["12110"]["parent"] == "121"
    assert by_code["12220"]["parent"] == "122"


def test_names_and_descriptions_are_stripped_and_missing_description_is_empty():
    by_code = {c["code"]: c for c in parse_purpose_sheet(SHEET)["codes"]}
    assert by_code["12110"]["name"] == "Health policy and administrative management"
    assert by_code["12110"]["description"] == "Health sector policy."
    assert by_code["12220"]["description"] == ""


def test_voluntary_column_sets_the_voluntary_flag():
    by_code = {c["code"]: c for c in parse_purpose_sheet(SHEET)["codes"]}
    assert by_code["12196"]["voluntary"] is True
    assert by_code["12110"]["voluntary"] is False


def test_three_digit_sector_headers_are_kept_with_their_names():
    sectors = parse_purpose_sheet(SHEET)["sectors"]
    assert sectors["121"] == "Health, General"
    assert sectors["120"] == "Health"
