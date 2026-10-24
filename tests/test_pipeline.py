"""End to end on the committed sample: baselines are measured, the LLM row is honest about not running."""

import json
import os
import re

import pytest

from src.data import load_splits
from src.run_pipeline import run_full_pipeline
from tests.fakes import FakeTextGenerator


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    for name in ("VERTEX_PROJECT", "VERTEX_REGION", "VERTEX_ENDPOINT_ID"):
        os.environ.pop(name, None)
    result = run_full_pipeline(results_dir=out / "results", figure_path=out / "figure.png")
    return result, out


def test_both_baselines_are_scored_on_the_whole_test_split(run):
    result, _ = run
    n_test = len(load_splits().test)
    for name in ("tfidf_logreg", "tfidf_knn"):
        assert result["baselines"][name]["test"]["n"] == n_test
        assert 0.5 < result["baselines"][name]["test"]["accuracy"] < 1.0


def test_threshold_is_chosen_on_val_and_reported_once_on_test(run):
    result, _ = run
    entry = result["baselines"]["tfidf_logreg"]["abstention"]
    assert entry["chosen_on"] == "val"
    assert set(entry) >= {"threshold", "target_precision", "val", "test"}
    assert set(entry["test"]) == {"coverage", "precision", "answered"}


def test_llm_row_says_it_was_not_run_when_no_endpoint_is_configured(run):
    result, _ = run
    assert result["llm"]["status"] == "not run"
    assert "no endpoint configured" in result["llm"]["reason"]
    assert "accuracy" not in result["llm"]


def test_metrics_json_matches_the_returned_result(run):
    result, out = run
    assert json.loads((out / "results" / "metrics.json").read_text()) == result


def test_sample_prompts_show_three_real_prompts_with_retrieved_examples(run):
    _, out = run
    text = (out / "results" / "sample_prompts.md").read_text()
    prompts = re.findall(r"```text\n(.*?)\n```", text, flags=re.S)
    assert len(prompts) == 3
    for prompt in prompts:
        assert prompt.endswith("Purpose code:")
        assert len(re.findall(r"^Purpose code: \d{5}$", prompt, flags=re.M)) == 5   # all k=5 examples fit
    test_texts = {r.text[:60] for r in load_splits().test}
    assert all(any(t in p for t in test_texts) for p in prompts)


def test_figure_is_written(run):
    _, out = run
    assert (out / "figure.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_configured_generator_gets_the_llm_row_measured(tmp_path):
    first_offered_code = lambda prompt: re.search(r"^(\d{5}) - ", prompt, flags=re.M).group(1)  # noqa: E731
    generator = FakeTextGenerator(first_offered_code)
    result = run_full_pipeline(results_dir=tmp_path, figure_path=tmp_path / "f.png", generator=generator)
    llm = result["llm"]
    assert llm["status"] == "run"
    assert llm["k"] == 5
    assert llm["test"]["n"] == len(load_splits().test)
    assert llm["parse_failures"] == 0
    assert len(generator.prompts) == llm["test"]["n"]
