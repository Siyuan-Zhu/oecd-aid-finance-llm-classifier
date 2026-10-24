"""build_prompt: what the model sees. Pure function, so the tests read the string."""

import pytest

from src.prompting import approx_tokens, build_prompt
from src.taxonomy import load_taxonomy
from tests.fakes import make_record


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


QUERY = make_record(0, "Query project about bed nets.", "12262")
NEAR = make_record(1, "Near example about malaria nets.", "12262")
FAR = make_record(2, "Far example about vaccination clinics.", "12250")


def test_prompt_lists_the_criteria_for_the_sectors_codes_only(taxonomy):
    prompt = build_prompt(QUERY, [NEAR], taxonomy)
    assert "12262" in prompt and "Malaria control" in prompt
    assert "Prevention and control of malaria." in prompt
    assert "Basic Health" in prompt
    assert "12110" not in prompt          # a code from sector 121
    assert "12340" not in prompt          # a code from sector 123


def test_voluntary_codes_are_left_out_unless_passed_as_candidates(taxonomy):
    query = make_record(0, "Query", "12110", "121")
    assert "12196" not in build_prompt(query, [], taxonomy)
    only = [taxonomy.get("12110"), taxonomy.get("12196")]
    assert "12196" in build_prompt(query, [], taxonomy, candidates=only)


def test_candidates_limit_which_codes_are_offered(taxonomy):
    prompt = build_prompt(QUERY, [], taxonomy, candidates=[taxonomy.get("12262"), taxonomy.get("12263")])
    assert "12262" in prompt and "12263" in prompt
    assert "12220" not in prompt


def test_examples_are_shown_least_similar_first_with_the_most_similar_next_to_the_query(taxonomy):
    prompt = build_prompt(QUERY, [NEAR, FAR], taxonomy)        # retriever order: most similar first
    assert prompt.index(FAR.text) < prompt.index(NEAR.text) < prompt.index(QUERY.text)


def test_examples_are_input_output_pairs_and_the_query_label_is_withheld(taxonomy):
    prompt = build_prompt(QUERY, [NEAR], taxonomy)
    assert f"Project: {NEAR.text}\nPurpose code: 12262\n" in prompt
    assert prompt.endswith(f"Project: {QUERY.text}\nPurpose code:")


def test_the_prompt_is_a_pure_function_of_its_inputs(taxonomy):
    assert build_prompt(QUERY, [NEAR, FAR], taxonomy) == build_prompt(QUERY, [NEAR, FAR], taxonomy)


def test_least_similar_examples_are_dropped_first_when_over_budget(taxonomy):
    words = lambda text: len(text.split())          # noqa: E731  (count by words so the budget is easy to set)
    full = build_prompt(QUERY, [NEAR, FAR], taxonomy, count_tokens=words)
    budget = words(full) - 1
    prompt = build_prompt(QUERY, [NEAR, FAR], taxonomy, max_tokens=budget, reserve_tokens=0, count_tokens=words)
    assert NEAR.text in prompt
    assert FAR.text not in prompt
    assert words(prompt) <= budget


def test_a_long_query_is_cut_to_fit_when_no_example_can_be_kept(taxonomy):
    long_query = make_record(0, " ".join(f"word{i}" for i in range(5000)))
    prompt = build_prompt(long_query, [NEAR], taxonomy, max_tokens=2048)
    assert approx_tokens(prompt) <= 2048 - 16
    assert NEAR.text not in prompt
    assert prompt.endswith("Purpose code:")
    assert "word0 " in prompt


def test_default_budget_is_flan_ul2s_2048_tokens(taxonomy):
    examples = [make_record(i, "Example text about malaria nets and spraying. " * 40) for i in range(1, 30)]
    prompt = build_prompt(QUERY, examples, taxonomy)
    assert approx_tokens(prompt) <= 2048 - 16
    assert 0 < prompt.count("Purpose code: 12262\n") < 29
