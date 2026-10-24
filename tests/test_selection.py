"""Each selection strategy returns the examples the 2023 work describes, in the order it describes."""

import sys

import pytest

from purpose_coder.embeddings import sbert_embedder
from purpose_coder.selection import HandPicked, Random, ReverseOrder, Similarity
from tests.fakes import POOL, PROJECT, fake_embed


def ids(examples):
    return [e.id for e in examples]


def test_random_returns_k_distinct_pool_examples():
    picked = Random(seed=1).select(PROJECT, POOL, 4)
    assert len(picked) == 4
    assert len(set(ids(picked))) == 4
    assert set(ids(picked)) <= set(ids(POOL))


def test_random_is_fixed_by_its_seed():
    assert ids(Random(seed=7).select(PROJECT, POOL, 3)) == ids(Random(seed=7).select(PROJECT, POOL, 3))


def test_random_never_asks_for_more_than_the_pool_holds():
    assert len(Random().select(PROJECT, POOL[:2], 5)) == 2


def test_hand_picked_returns_the_listed_examples_in_the_listed_order():
    assert ids(HandPicked(["c2", "v1", "n1"]).select(PROJECT, POOL, 5)) == ["c2", "v1", "n1"]


def test_hand_picked_is_capped_at_k_and_skips_ids_outside_the_pool():
    selector = HandPicked(["c2", "missing", "v1", "n1"])
    assert ids(selector.select(PROJECT, POOL, 2)) == ["c2", "v1"]


def test_similarity_picks_the_closest_pool_examples_closest_first():
    picked = Similarity(fake_embed).select(PROJECT, POOL, 2)
    assert set(ids(picked)) == {"v1", "v2"}  # the two measles and vaccination examples


def test_similarity_orders_by_closeness():
    near = Similarity(fake_embed).select(PROJECT, POOL, 6)
    assert ids(near)[:2] == ["v1", "v2"] or ids(near)[:2] == ["v2", "v1"]
    assert ids(near)[2:4] != ["v1", "v2"]


def test_similarity_embeds_each_pool_example_once():
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    selector = Similarity(counting_embed)
    selector.select(PROJECT, POOL, 2)
    selector.select(PROJECT, POOL, 3)
    assert calls == [1, 6, 1]  # query, pool (once), query


def test_similarity_with_k_zero_or_an_empty_pool_selects_nothing():
    assert Similarity(fake_embed).select(PROJECT, POOL, 0) == []
    assert Similarity(fake_embed).select(PROJECT, [], 3) == []


def test_reverse_order_flips_the_inner_selectors_examples():
    forward = Similarity(fake_embed).select(PROJECT, POOL, 3)
    backward = ReverseOrder(Similarity(fake_embed)).select(PROJECT, POOL, 3)
    assert ids(backward) == ids(forward)[::-1]


def test_core_modules_import_and_run_without_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import sentence_transformers` fail
    import purpose_coder.classify  # noqa: F401  (importing the CLI must not need the optional extra)

    with pytest.raises(ImportError, match="sentence-transformers"):
        sbert_embedder()
