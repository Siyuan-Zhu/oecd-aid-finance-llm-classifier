"""Retrievers return the labelled examples most similar to a record, most similar first."""

import sys

import pytest

from src.retrieval import SbertRetriever, TfidfRetriever
from tests.fakes import make_record


POOL = [
    make_record(1, "malaria bed nets distribution insecticide treated", "12262"),
    make_record(2, "tuberculosis treatment drugs clinics detection", "12263"),
    make_record(3, "malaria prevention spraying and bed nets in villages", "12262"),
    make_record(4, "nutrition supplements for children", "12240"),
    make_record(5, "malaria bed nets for solar mini grids", "23231", code3="232"),
]


def test_top_k_returns_most_similar_first():
    query = make_record(10, "distribution of insecticide treated bed nets against malaria", "12262")
    top = TfidfRetriever(POOL).top_k(query, 2)
    assert [r.id for r in top] == ["1", "3"]


def test_only_examples_from_the_same_three_digit_sector_are_returned():
    query = make_record(10, "malaria bed nets", "12262")
    top = TfidfRetriever(POOL).top_k(query, 10)
    assert {r.code3 for r in top} == {"122"}
    assert "5" not in {r.id for r in top}


def test_the_query_record_itself_is_never_returned():
    top = TfidfRetriever(POOL).top_k(POOL[0], 3)
    assert POOL[0].id not in {r.id for r in top}


def test_k_larger_than_the_pool_returns_what_there_is():
    query = make_record(10, "malaria", "12262")
    assert len(TfidfRetriever(POOL).top_k(query, 50)) == 4


def test_sbert_retriever_explains_the_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        SbertRetriever(POOL)
