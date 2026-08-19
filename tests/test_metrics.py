from longspeech_eval.metrics import (
    asr_metrics,
    character_error_rate,
    emotion_metrics,
    language_detection_metrics,
    numeric_fixed_answer_metrics,
    temporal_judge_metrics,
    translation_metrics,
)
from longspeech_eval.temporal_judge import parse_judgment


def rec(ref, pred, *, language="en"):
    return {"reference": ref, "prediction": pred, "language": language, "error": None}


def test_cer_identity():
    assert character_error_rate("你好 world", "你好 world") == 0.0


def test_asr_cjk_group_uses_dataset_language_not_transcript_characters():
    # Korean example deliberately uses Latin text; paper grouping is by language.
    rows = [
        rec("annyeong haseyo", "annyeong haseyo", language="ko"),
        rec("hello world", "hello world", language="en"),
    ]
    m = asr_metrics(rows)
    assert m["cjk_n"] == 1
    assert m["non_cjk_n"] == 1


def test_numeric_parser_matches_fixed_answer_relations():
    m = numeric_fixed_answer_metrics([rec("2", "There are 2 speakers."), rec("3", "unknown")])
    assert m["numeric_accuracy"] == 50.0
    assert m["parsability_rate"] == 50.0
    assert m["post_parsing_precision"] == 100.0
    assert m["misunderstanding_rate"] == 50.0


def test_emotion_exact_fine_to_coarse_mapping():
    m = emotion_metrics([
        rec("Excited", "Amused"),                       # same coarse: strict
        rec("Hopeful", "Determined"),                  # same coarse: strict
        rec("Melancholy", "Reflective"),               # same coarse: strict
        rec("Suspenseful", "Serious"),                 # same coarse: strict
        rec("Intrigued", "Questioning"),               # same coarse: strict
        rec("Confused", "Cognitive-Uncertainty"),      # fine -> direct coarse
        rec("Neutral", "Calm"),                        # same coarse: strict
    ])
    assert m["strict_accuracy"] == 100.0
    assert m["relaxed_accuracy"] == 100.0


def test_emotion_relaxed_polarity():
    m = emotion_metrics([
        rec("Excited", "Hopeful"),          # positive subtype differs
        rec("Suspenseful", "Melancholy"),  # negative subtype differs
        rec("Intrigued", "Confused"),       # cognitive subtype differs
        rec("Neutral", "Neutral"),
    ])
    assert m["strict_accuracy"] == 25.0
    assert m["relaxed_accuracy"] == 100.0


def test_language_detection_three_way_partition():
    rows = [
        rec("English", "The language is English.", language="en"),
        rec("French", "German", language="fr"),
        rec("German", "I cannot determine it.", language="de"),
    ]
    m = language_detection_metrics(rows)
    assert abs(m["detection_accuracy"] - 100 / 3) < 1e-12
    assert abs(m["misunderstanding_rate"] - 100 / 3) < 1e-12
    assert abs(m["detection_errors"] - 100 / 3) < 1e-12
    assert abs(m["detection_accuracy"] + m["misunderstanding_rate"] + m["detection_errors"] - 100.0) < 1e-12


def test_temporal_judgment_metrics():
    rows = [
        {"judgment": "YES", "judge_error": None},
        {"judgment": "PARTIALLY", "judge_error": None},
        {"judgment": "NO", "judge_error": None},
        {"judgment": "YES", "judge_error": None},
        {"judgment": None, "judge_error": "timeout"},
    ]
    m = temporal_judge_metrics(rows)
    assert m["strict_accuracy"] == 50.0
    assert m["relaxed_accuracy"] == 75.0
    assert m["n"] == 4
    assert m["judge_rows"] == 5
    assert m["judge_errors"] == 1
    assert m["judge_valid_coverage_pct"] == 80.0


def test_temporal_parser():
    assert parse_judgment("YES") == "YES"
    assert parse_judgment("PARTIALLY.") == "PARTIALLY"
    assert parse_judgment("NO - incorrect") == "NO"
    assert parse_judgment("maybe") is None


def test_bleu_perfect():
    import pytest
    pytest.importorskip("sacrebleu")
    m = translation_metrics([rec("hello world", "hello world")])
    assert m["bleu4_case_insensitive"] > 99.0
