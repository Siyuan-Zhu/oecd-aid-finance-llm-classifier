"""classify_projects and the CLI, driven by a fake generator and fake embeddings."""

import csv

import pytest

from purpose_coder.classify import OK, UNKNOWN_SECTOR, UNPARSED, classify_projects, main, make_selector
from purpose_coder.data import Project
from purpose_coder.selection import Similarity
from purpose_coder.taxonomy import load_taxonomy
from tests.fakes import POOL, PROJECT, FakeGenerator, fake_embed


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


def test_a_valid_answer_becomes_the_prediction(taxonomy):
    generator = FakeGenerator("12250")
    (prediction,) = classify_projects([PROJECT], POOL, generator, Similarity(fake_embed), taxonomy, k=2)
    assert (prediction.code5, prediction.status) == ("12250", OK)
    assert set(prediction.example_ids) == {"v1", "v2"}


def test_each_prompt_holds_k_examples_from_the_projects_own_sector(taxonomy):
    other_sector = Project("Clean water", "Drills wells for villages.", "140")
    generator = FakeGenerator("12250")
    classify_projects([PROJECT, other_sector], POOL, generator, Similarity(fake_embed), taxonomy, k=3)
    assert generator.prompts[0].count("Answer: 12") == 3
    assert "Answer:" not in generator.prompts[1].split("New:")[0]  # no sector-140 examples in the pool: zero-shot
    assert "14010" in generator.prompts[1]  # water guidelines, not health ones


def test_invalid_answers_are_marked_unparsed_with_no_code(taxonomy):
    generator = FakeGenerator(["banana", "12110", "12250 12230", "99999"])
    projects = [PROJECT] * 4
    predictions = classify_projects(projects, POOL, generator, Similarity(fake_embed), taxonomy, k=1)
    assert [p.status for p in predictions] == [UNPARSED] * 4
    assert [p.code5 for p in predictions] == [None] * 4
    assert predictions[0].raw_output == "banana"


def test_unknown_sector_is_not_sent_to_the_model(taxonomy):
    generator = FakeGenerator("12250")
    (prediction,) = classify_projects([Project("t", "d", "999")], POOL, generator, Similarity(fake_embed), taxonomy)
    assert prediction.status == UNKNOWN_SECTOR
    assert generator.prompts == []


def test_k_zero_is_zero_shot(taxonomy):
    generator = FakeGenerator("12250")
    classify_projects([PROJECT], POOL, generator, Similarity(fake_embed), taxonomy, k=0)
    assert "Answer:" not in generator.prompts[0]


def test_hand_picked_ids_must_exist_in_the_pool():
    with pytest.raises(ValueError, match="nope"):
        make_selector("hand-picked", POOL, ids=["v1", "nope"])


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_cli_writes_predictions_and_keeps_input_columns(tmp_path, capsys):
    write_csv(tmp_path / "in.csv", ["ref", "title", "description", "code3"],
              [["a", PROJECT.title, PROJECT.description, "122"], ["b", "Odd", "Unclear text.", "122"]])
    write_csv(tmp_path / "pool.csv", ["id", "title", "description", "code5"],
              [[e.id, e.title, e.description, e.code5] for e in POOL])
    out = tmp_path / "out.csv"
    generator = FakeGenerator(["12250", "no idea"])
    code = main(
        ["--unlabelled", str(tmp_path / "in.csv"), "--pool", str(tmp_path / "pool.csv"), "--endpoint", "fake",
         "--output", str(out), "--k", "2", "--reverse"],
        generator=generator, embed=fake_embed,
    )
    assert code == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["ref"] for r in rows] == ["a", "b"]
    assert [r["predicted_code5"] for r in rows] == ["12250", ""]
    assert [r["status"] for r in rows] == [OK, UNPARSED]
    assert rows[0]["example_ids"] == "v2;v1"  # reversed: the closest example comes last
    assert "1 unparsed" in capsys.readouterr().out


def test_cli_rejects_a_pool_row_with_a_bad_code(tmp_path):
    write_csv(tmp_path / "in.csv", ["title", "description", "code3"], [["a", "b", "122"]])
    write_csv(tmp_path / "pool.csv", ["title", "description", "code5"], [["a", "b", "122"]])
    with pytest.raises(ValueError, match="5 digits"):
        main(["--unlabelled", str(tmp_path / "in.csv"), "--pool", str(tmp_path / "pool.csv"), "--endpoint", "x",
              "--output", str(tmp_path / "o.csv")], generator=FakeGenerator("1"), embed=fake_embed)
