"""The taxonomy answers questions about the official OECD purpose-code list."""

import pytest

from purpose_coder.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


def test_solar_codes_carry_the_names_oecd_gives_them(taxonomy):
    assert taxonomy.name("23230") == "Solar energy for centralised grids"
    assert taxonomy.name("23231") == "Solar energy for isolated grids and standalone systems"


def test_children_of_basic_health_are_the_nine_core_codes(taxonomy):
    codes = [c.code for c in taxonomy.children("122")]
    assert codes == ["12220", "12230", "12240", "12250", "12261", "12262", "12263", "12264", "12281"]


def test_children_carry_name_and_description(taxonomy):
    malaria = next(c for c in taxonomy.children("122") if c.code == "12262")
    assert malaria.name == "Malaria control"
    assert malaria.description == "Prevention and control of malaria."


def test_voluntary_codes_are_children_and_flagged(taxonomy):
    stats = next(c for c in taxonomy.children("121") if c.code == "12196")
    assert stats.voluntary is True


def test_parent_of_a_five_digit_code_is_its_three_digit_sector(taxonomy):
    assert taxonomy.parent("12262") == "122"
    assert taxonomy.parent("23231") == "232"


def test_three_digit_sector_has_a_name(taxonomy):
    assert taxonomy.name("123") == "Non-communicable diseases (NCDs)"


def test_unknown_codes_raise_key_error(taxonomy):
    with pytest.raises(KeyError):
        taxonomy.parent("99999")
    with pytest.raises(KeyError):
        taxonomy.children("999")
