"""Classifiers: the TF-IDF baselines and the few-shot LLM adapter, all through fit/predict."""

import pytest

from src.classifiers import FewShotLLM, TfidfKnn, TfidfLogReg
from src.retrieval import TfidfRetriever
from src.taxonomy import load_taxonomy
from tests.fakes import FakeTextGenerator, make_record


MALARIA = ["malaria bed nets insecticide mosquito", "malaria spraying mosquito nets villages", "anti-malarial drugs mosquito"]
TB = ["tuberculosis treatment clinics detection", "tuberculosis drugs patients sputum", "tb screening clinics treatment"]
MENTAL = ["mental health counselling psychosocial", "psychosocial support depression counselling"]
TRAIN = (
    [make_record(i, t, "12262") for i, t in enumerate(MALARIA)]
    + [make_record(10 + i, t, "12263") for i, t in enumerate(TB)]
    + [make_record(20 + i, t, "12340") for i, t in enumerate(MENTAL)]
)
CHILDREN_122 = {c.code for c in load_taxonomy().children("122")}


@pytest.fixture(params=[TfidfLogReg, TfidfKnn], ids=["logreg", "knn"])
def baseline(request):
    return request.param().fit(TRAIN)


def test_fit_returns_the_classifier(baseline):
    assert baseline.fit(TRAIN) is baseline


def test_baselines_learn_a_separable_toy_problem(baseline):
    queries = [make_record(100, "bed nets against mosquito", "12262"), make_record(101, "tuberculosis clinics", "12263"),
               make_record(102, "counselling for depression", "12340")]
    assert [p.code5 for p in baseline.predict(queries)] == ["12262", "12263", "12340"]


def test_predictions_never_leave_the_records_sector(baseline):
    # text that looks like mental health, but the record is in sector 122: only 122 codes are allowed
    query = make_record(100, "counselling for depression psychosocial", "12262")
    prediction = baseline.predict([query])[0]
    assert prediction.code5 in CHILDREN_122


def test_confidence_is_a_probability_like_number(baseline):
    for p in baseline.predict([make_record(100, "bed nets", "12262"), make_record(101, "unrelated words entirely", "12263")]):
        assert 0.0 <= p.confidence <= 1.0


def test_a_sector_with_no_training_data_abstains(baseline):
    prediction = baseline.predict([make_record(100, "solar panels", "23230", code3="232")])[0]
    assert prediction.code5 is None
    assert prediction.confidence == 0.0


def test_a_sector_with_a_single_training_code_always_answers_it():
    train = [make_record(1, "alcohol harm reduction", "12330"), make_record(2, "drug abuse prevention", "12330")]
    for model in (TfidfLogReg(), TfidfKnn()):
        model.fit(train)
        assert model.predict([make_record(3, "anything", "12330")])[0].code5 == "12330"


def test_knn_is_more_confident_when_neighbours_agree():
    knn = TfidfKnn(k=3).fit(TRAIN)
    agree = knn.predict([make_record(100, "malaria mosquito nets", "12262")])[0]
    split = knn.predict([make_record(101, "malaria tuberculosis clinics mosquito", "12262")])[0]
    assert agree.confidence > split.confidence


# ---- few-shot LLM ----

def few_shot(answers, k=2):
    generator = FakeTextGenerator(answers)
    return FewShotLLM(generator, TfidfRetriever, k=k), generator


def test_the_prompt_contains_the_sectors_criteria_and_the_nearest_training_examples():
    model, generator = few_shot(["12262"])
    model.fit(TRAIN)
    model.predict([make_record(100, "bed nets against malaria mosquito", "12262")])
    prompt = generator.prompts[0]
    assert "Malaria control" in prompt and "Tuberculosis control" in prompt
    assert "Mental health" not in prompt                    # sector 123 criteria stay out
    assert MALARIA[0] in prompt                             # nearest labelled example is shown
    assert prompt.endswith("Purpose code:")


def test_a_valid_answer_becomes_the_prediction():
    model, _ = few_shot(["12263"])
    model.fit(TRAIN)
    prediction = model.predict([make_record(100, "anything", "12262")])[0]
    assert prediction.code5 == "12263"
    assert model.parse_failures == 0


def test_the_first_five_digit_number_in_chatty_output_is_used():
    model, _ = few_shot(["The purpose code is 12262."])
    model.fit(TRAIN)
    assert model.predict([make_record(100, "anything", "12262")])[0].code5 == "12262"


@pytest.mark.parametrize("answer", ["", "malaria", "12", "12110", "99999", "123456789"])
def test_output_that_is_not_a_valid_child_code_abstains_and_is_counted(answer):
    model, _ = few_shot([answer])
    model.fit(TRAIN)
    prediction = model.predict([make_record(100, "anything", "12262")])[0]
    assert prediction.code5 is None
    assert model.parse_failures == 1


def test_a_code_the_training_data_never_uses_is_not_a_valid_answer():
    model, _ = few_shot(["12220"])          # a real child of sector 122, but absent from TRAIN
    model.fit(TRAIN)
    assert model.predict([make_record(100, "anything", "12262")])[0].code5 is None


def test_one_answer_per_record_in_order():
    model, generator = few_shot(["12262", "12263", "12340"])
    model.fit(TRAIN)
    records = [make_record(100, "a", "12262"), make_record(101, "b", "12263"), make_record(102, "c", "12340")]
    assert [p.code5 for p in model.predict(records)] == ["12262", "12263", "12340"]
    assert len(generator.prompts) == 3
