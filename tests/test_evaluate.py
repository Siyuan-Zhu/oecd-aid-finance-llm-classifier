"""evaluate: accuracy, macro-F1, per-sector accuracy, confusion pairs, and the abstention threshold."""

import pytest

from src.classifiers import Prediction
from src.data import Record
from src.evaluate import apply_threshold, choose_threshold, evaluate, selective


def rec(i, code5, code3=None):
    return Record(id=str(i), text=f"text {i}", code3=code3 or code5[:3], code5=code5, year=2024, donor="D")


def pred(code5, confidence=1.0):
    return Prediction(code5=code5, confidence=confidence)


# Truth: A A A B B C (A=12262, B=12263 in sector 122; C=12340 in sector 123)
RECORDS = [rec(1, "12262"), rec(2, "12262"), rec(3, "12262"), rec(4, "12263"), rec(5, "12263"), rec(6, "12340")]


def test_accuracy_counts_abstentions_as_wrong():
    preds = [pred("12262"), pred("12262"), pred(None, 0.0), pred("12263"), pred("12262"), pred("12340")]
    metrics = evaluate(preds, RECORDS)
    assert metrics.n == 6
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.abstentions == 1


def test_macro_f1_is_the_mean_of_per_code_f1():
    # 12262: tp 2, fp 1, fn 1 -> F1 4/6 ; 12263: tp 1, fp 1, fn 1 -> F1 2/4 ; 12340: tp 1 -> F1 1
    preds = [pred("12262"), pred("12262"), pred("12263"), pred("12263"), pred("12262"), pred("12340")]
    metrics = evaluate(preds, RECORDS)
    assert metrics.macro_f1 == pytest.approx((4 / 6 + 2 / 4 + 1) / 3)


def test_per_sector_accuracy_uses_the_three_digit_code():
    preds = [pred("12262"), pred("12263"), pred("12262"), pred("12263"), pred("12263"), pred("12340")]
    metrics = evaluate(preds, RECORDS)
    assert metrics.per_sector == {"122": pytest.approx(4 / 5), "123": pytest.approx(1.0)}


def test_per_code_accuracy_is_recall_of_each_true_code():
    preds = [pred("12262"), pred("12263"), pred("12262"), pred("12263"), pred("12262"), pred("12340")]
    metrics = evaluate(preds, RECORDS)
    assert metrics.per_code == {"12262": pytest.approx(2 / 3), "12263": pytest.approx(1 / 2), "12340": 1.0}


def test_confusion_pairs_list_the_most_frequent_mistakes_first():
    preds = [pred("12263"), pred("12263"), pred("12262"), pred("12262"), pred("12263"), pred("12340")]
    metrics = evaluate(preds, RECORDS)
    assert metrics.confusions[0] == ("12262", "12263", 2)
    assert ("12263", "12262", 1) in metrics.confusions


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        evaluate([pred("12262")], RECORDS)


def test_apply_threshold_abstains_below_the_threshold():
    out = apply_threshold([pred("12262", 0.9), pred("12263", 0.4)], 0.5)
    assert [p.code5 for p in out] == ["12262", None]


def test_selective_reports_coverage_and_precision_among_answered():
    preds = [pred("12262", 0.9), pred("12263", 0.9), pred("12263", 0.9), pred(None, 0.0), pred("12263", 0.9), pred("12340", 0.9)]
    result = selective(preds, RECORDS)
    assert result.answered == 5
    assert result.coverage == pytest.approx(5 / 6)
    # truth       12262  12262  12262  12263  12263  12340
    # prediction  12262  12263  12263  None   12263  12340   -> right, wrong, wrong, (abstains), right, right
    assert result.precision == pytest.approx(3 / 5)


def test_choose_threshold_takes_the_widest_coverage_that_meets_the_precision_target():
    # confidence rises with correctness: wrong answers are the low-confidence ones
    preds = [pred("12262", 0.95), pred("12262", 0.9), pred("12263", 0.3), pred("12263", 0.85), pred("12262", 0.2), pred("12340", 0.8)]
    # truth: 12262 12262 12262 12263 12263 12340 -> right, right, wrong, right, wrong, right
    threshold = choose_threshold(preds, RECORDS, target_precision=1.0)
    assert threshold == pytest.approx(0.8)
    kept = selective(apply_threshold(preds, threshold), RECORDS)
    assert kept.precision == 1.0
    assert kept.coverage == pytest.approx(4 / 6)


def test_choose_threshold_falls_back_to_answering_everything_when_no_threshold_meets_the_target():
    preds = [pred("12263", 0.9)] * 6
    assert choose_threshold(preds, RECORDS, target_precision=0.99) == 0.0
