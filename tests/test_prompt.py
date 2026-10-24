"""build_prompt lays out the 2023 template; parse_answer accepts only a valid 5-digit child code."""

import pytest

from purpose_coder.data import Project
from purpose_coder.prompt import INSTRUCTION, build_prompt, format_guidelines, parse_answer
from purpose_coder.taxonomy import load_taxonomy
from tests.fakes import POOL, PROJECT


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


def test_guidelines_list_every_five_digit_code_of_the_sector_with_its_clarification(taxonomy):
    guidelines = format_guidelines("122", taxonomy)
    lines = guidelines.split("\n\n")
    assert [line.split()[0] for line in lines] == [c.code for c in taxonomy.children("122")]
    assert "12262 Malaria control; clarification: Prevention and control of malaria." in lines


def test_guidelines_hold_only_the_projects_own_sector(taxonomy):
    guidelines = format_guidelines("122", taxonomy)
    assert "12110" not in guidelines  # a 121 code
    assert "12230" in guidelines


def test_prompt_follows_the_2023_layout(taxonomy):
    prompt = build_prompt(PROJECT, POOL[:2], taxonomy)
    blocks = prompt.split("\n\n")
    assert blocks[0] == INSTRUCTION
    assert blocks[1].startswith("Criteria and clarifications:")
    assert blocks[-3].startswith("input:(title and description): " + POOL[0].text)
    assert blocks[-3].endswith("Answer: 12250")
    assert blocks[-2].startswith("input:(title and description): " + POOL[1].text)
    assert blocks[-1] == f"New:\ninput(title and description): {PROJECT.text}\nanswer:"


def test_title_and_description_are_joined_with_a_semicolon(taxonomy):
    assert PROJECT.text == "Measles campaign ; Vaccinate children against measles across several districts."


def test_examples_appear_in_the_order_given(taxonomy):
    prompt = build_prompt(PROJECT, [POOL[3], POOL[0]], taxonomy)
    assert prompt.index(POOL[3].title) < prompt.index(POOL[0].title)


def test_no_examples_gives_the_zero_shot_prompt(taxonomy):
    prompt = build_prompt(PROJECT, [], taxonomy)
    assert "Answer:" not in prompt
    assert prompt.endswith(f"New:\ninput(title and description): {PROJECT.text}\nanswer:")


def test_unknown_sector_has_no_guidelines(taxonomy):
    with pytest.raises(KeyError):
        build_prompt(Project("t", "d", "999"), [], taxonomy)


@pytest.mark.parametrize("output", ["12250", " 12250\n", "Answer: 12250", "12250 Infectious disease control"])
def test_a_valid_child_code_is_parsed(taxonomy, output):
    assert parse_answer(output, "122", taxonomy) == "12250"


@pytest.mark.parametrize(
    "output",
    [
        "",
        "I do not know",
        "1225",           # too short
        "122501",         # too long
        "99999",          # not an OECD code
        "12110",          # a real code, but from sector 121
        "12250 or 12230",  # two codes
    ],
)
def test_anything_else_is_rejected(taxonomy, output):
    assert parse_answer(output, "122", taxonomy) is None


def test_unknown_sector_rejects_every_answer(taxonomy):
    assert parse_answer("12250", "999", taxonomy) is None
