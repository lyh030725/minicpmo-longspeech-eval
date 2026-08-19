from __future__ import annotations

import re
import unicodedata
from collections import Counter


CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)

# Common aliases are used as a fallback. For the full benchmark, language labels
# are also learned from the test references themselves so uncommon languages are
# not limited to this table.
LANG_NAMES = {
    "en": {"en", "eng", "english"},
    "zh": {"zh", "zho", "chi", "chinese", "mandarin", "中文", "汉语", "漢語"},
    "de": {"de", "deu", "ger", "german", "deutsch"},
    "fr": {"fr", "fra", "fre", "french", "français", "francais"},
    "it": {"it", "ita", "italian", "italiano"},
    "es": {"es", "spa", "spanish", "español", "espanol"},
    "ja": {"ja", "jpn", "japanese", "日本語"},
    "ko": {"ko", "kor", "korean", "한국어"},
    "ru": {"ru", "rus", "russian", "русский"},
    "pt": {"pt", "por", "portuguese", "português", "portugues"},
    "nl": {"nl", "nld", "dut", "dutch", "nederlands"},
    "pl": {"pl", "pol", "polish", "polski"},
    "cs": {"cs", "ces", "cze", "czech", "čeština", "cestina"},
    "sv": {"sv", "swe", "swedish", "svenska"},
    "fi": {"fi", "fin", "finnish", "suomi"},
    "da": {"da", "dan", "danish", "dansk"},
    "no": {"no", "nor", "norwegian", "norsk"},
    "ro": {"ro", "ron", "rum", "romanian", "română", "romana"},
    "tr": {"tr", "tur", "turkish", "türkçe", "turkce"},
    "ar": {"ar", "ara", "arabic", "العربية"},
    "hi": {"hi", "hin", "hindi", "हिन्दी", "हिंदी"},
    "id": {"id", "ind", "indonesian", "bahasa indonesia"},
    "vi": {"vi", "vie", "vietnamese", "tiếng việt", "tieng viet"},
    "uk": {"uk", "ukr", "ukrainian", "українська"},
    "el": {"el", "ell", "gre", "greek", "ελληνικά"},
    "hu": {"hu", "hun", "hungarian", "magyar"},
}

# Exact fine -> coarse mapping from LongSpeech Sec. 3.2.2.
EMOTION_FINE_TO_COARSE = {
    "excited": "positive-higharousal",
    "amused": "positive-higharousal",
    "hopeful": "positive-uplifting",
    "determined": "positive-uplifting",
    "inspirational": "positive-uplifting",
    "awe": "positive-uplifting",
    "vivid": "positive-uplifting",
    "melancholy": "negative-sadness/reflection",
    "somber": "negative-sadness/reflection",
    "pensive": "negative-sadness/reflection",
    "reflective": "negative-sadness/reflection",
    "suspenseful": "negative-tension/seriousness",
    "serious": "negative-tension/seriousness",
    "dramatic": "negative-tension/seriousness",
    "urgent": "negative-tension/seriousness",
    "intense": "negative-tension/seriousness",
    "intrigued": "cognitive-curiosity",
    "questioning": "cognitive-curiosity",
    "confused": "cognitive-uncertainty",
    "neutral": "neutral-informative",
    "calm": "neutral-informative",
    "objective": "neutral-informative",
    "informative": "neutral-informative",
    "assertive": "neutral-informative",
}

EMOTION_COARSE_TO_POLARITY = {
    "positive-higharousal": "positive",
    "positive-uplifting": "positive",
    "negative-sadness/reflection": "negative",
    "negative-tension/seriousness": "negative",
    "cognitive-curiosity": "cognitive",
    "cognitive-uncertainty": "cognitive",
    "neutral-informative": "neutral",
}

TEMPORAL_JUDGMENTS = {"YES", "PARTIALLY", "NO"}


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def contains_cjk(text: str) -> bool:
    return any(_is_cjk_char(ch) for ch in text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_label(text: str) -> str:
    text = normalize_text(text).replace("_", "-")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (r != h),
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    ref = [c for c in normalize_text(reference) if not c.isspace()]
    hyp = [c for c in normalize_text(prediction) if not c.isspace()]
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _is_cjk_language(record: dict) -> bool:
    """Return whether a row belongs to the paper's CJK group (zh/ja/ko).

    LongSpeech defines CJK by language (Chinese/Japanese/Korean), not by whether
    a transcript happens to contain a CJK Unicode character. Use the dataset
    language field when available and only fall back to the transcript for old
    or externally-produced result files that lack that field.
    """
    language = normalize_text(record.get("language", "")).replace("_", "-")
    if language:
        first = language.split("-", 1)[0]
        if first in {"zh", "zho", "chi", "cmn", "ja", "jpn", "ko", "kor"}:
            return True
        if language in {"chinese", "mandarin", "japanese", "korean", "中文", "汉语", "漢語", "日本語", "한국어"}:
            return True
        return False
    return contains_cjk(record.get("reference", ""))


def asr_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    cjk = [r for r in ok if _is_cjk_language(r)]
    non_cjk = [r for r in ok if not _is_cjk_language(r)]

    def corpus_cer(rows):
        total_errors = 0
        total_chars = 0
        for r in rows:
            ref_chars = [c for c in normalize_text(r["reference"]) if not c.isspace()]
            hyp_chars = [c for c in normalize_text(r["prediction"]) if not c.isspace()]
            total_errors += _levenshtein(ref_chars, hyp_chars)
            total_chars += len(ref_chars)
        return (total_errors / total_chars) if total_chars else None

    non_cjk_wer = None
    if non_cjk:
        total_errors = 0
        total_words = 0
        for r in non_cjk:
            ref_words = normalize_text(r["reference"]).split()
            hyp_words = normalize_text(r["prediction"]).split()
            total_errors += _levenshtein(ref_words, hyp_words)
            total_words += len(ref_words)
        non_cjk_wer = (total_errors / total_words) if total_words else None

    return {
        "n": len(ok),
        "non_cjk_n": len(non_cjk),
        "cjk_n": len(cjk),
        "non_cjk_wer": non_cjk_wer,
        "cjk_cer": corpus_cer(cjk),
        "overall_cer": corpus_cer(ok),
        "note": "Paper-aligned metric families; text normalization may differ from the unreleased author evaluator.",
    }


def translation_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    if not ok:
        return {"n": 0, "bleu4_case_insensitive": None}
    refs = [r["reference"] for r in ok]
    hyps = [r["prediction"] for r in ok]
    from sacrebleu import corpus_bleu

    score = corpus_bleu(hyps, [refs], lowercase=True)
    return {"n": len(ok), "bleu4_case_insensitive": float(score.score)}


def summary_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = Counter()
    for r in ok:
        scores = scorer.score(r["reference"], r["prediction"])
        for name in ("rouge1", "rouge2", "rougeL"):
            sums[name] += scores[name].fmeasure
    n = len(ok)
    return {
        "n": n,
        "rouge1_f1": 100.0 * sums["rouge1"] / n if n else None,
        "rouge2_f1": 100.0 * sums["rouge2"] / n if n else None,
        "rougeL_f1": 100.0 * sums["rougeL"] / n if n else None,
        "unit": "percent (paper Table 4 scale)",
    }


def extract_first_integer(text: str) -> int | None:
    match = re.search(r"(?<!\d)-?\d+(?!\d)", str(text))
    return int(match.group(0)) if match else None


def numeric_fixed_answer_metrics(records: list[dict]) -> dict:
    """LongSpeech fixed-answer protocol inferred from Sec. 3.2.1/Table 4.

    The paper defines Numeric Accuracy, Parsability Rate, Post-Parsing Precision,
    and Misunderstanding Rate, but does not release its parsing code. Table 4 is
    internally consistent with M.R = 1 - P.R and Pr = N.A / P.R. We therefore
    treat a response as parsable when a numeric answer can be extracted.
    """
    ok = [r for r in records if r.get("error") is None]
    parsed = 0
    correct = 0
    reference_parsed = 0
    for r in ok:
        ref = extract_first_integer(r["reference"])
        pred = extract_first_integer(r["prediction"])
        if ref is not None:
            reference_parsed += 1
        if pred is not None:
            parsed += 1
        if ref is not None and pred == ref:
            correct += 1
    n = len(ok)
    parsability = parsed / n if n else None
    return {
        "n": n,
        "reference_numeric_n": reference_parsed,
        "numeric_accuracy": 100.0 * correct / n if n else None,
        "parsability_rate": 100.0 * parsability if parsability is not None else None,
        "post_parsing_precision": 100.0 * correct / parsed if parsed else None,
        "misunderstanding_rate": 100.0 * (1.0 - parsability) if parsability is not None else None,
        "unit": "percent (paper Table 4 scale)",
        "note": "Paper metric formulas reproduced; numeric parser is reconstructed because the authors did not release evaluator code.",
    }


def _language_alias_table(records: list[dict]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for code, names in LANG_NAMES.items():
        for name in names:
            aliases[_normalize_label(name)] = code

    # Learn the benchmark's own reference spellings. This makes the evaluator
    # cover every language present in the current result set, even when it is not
    # listed in LANG_NAMES.
    for r in records:
        code = normalize_text(r.get("language", ""))
        ref = _normalize_label(r.get("reference", ""))
        if code and ref and len(ref) <= 80:
            aliases[ref] = code
    return aliases


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    # ISO-like short codes must be token-delimited; language names can be safely
    # matched as normalized phrases.
    if re.fullmatch(r"[a-z]{2,3}", alias):
        return re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) is not None
    return alias in text


def _extract_language(text: str, aliases: dict[str, str]) -> str | None:
    normalized = _normalize_label(text)
    tag = re.search(r"<\|([a-z]{2,3})\|>", normalized)
    if tag:
        return tag.group(1)

    # Exact answer first, then longest aliases to avoid short-name collisions.
    if normalized in aliases:
        return aliases[normalized]
    for alias in sorted(aliases, key=len, reverse=True):
        # Do not treat ISO codes such as "it" as words inside free-form prose
        # (e.g. "I cannot determine it"). Short codes are accepted only as
        # exact answers above or in <|xx|> tags.
        if re.fullmatch(r"[a-z]{2,3}", alias):
            continue
        if _contains_alias(normalized, alias):
            return aliases[alias]
    return None


def language_detection_metrics(records: list[dict]) -> dict:
    """Reproduce Table-4 Language Detection columns: D.A, M.R, D.E.

    Table 4 shows these three percentages partitioning the examples (summing to
    100%). We interpret them as correct recognized language, recognized-but-wrong
    language (misunderstanding), and no parsable language label (detection error).
    """
    ok = [r for r in records if r.get("error") is None]
    aliases = _language_alias_table(ok)
    correct = wrong = detection_errors = 0
    for r in ok:
        expected = normalize_text(r.get("language", "")) or _extract_language(r["reference"], aliases)
        pred = _extract_language(r["prediction"], aliases)
        if pred is None:
            detection_errors += 1
        elif expected and pred == expected:
            correct += 1
        else:
            wrong += 1
    n = len(ok)
    return {
        "n": n,
        "detection_accuracy": 100.0 * correct / n if n else None,
        "misunderstanding_rate": 100.0 * wrong / n if n else None,
        "detection_errors": 100.0 * detection_errors / n if n else None,
        "parsed_n": correct + wrong,
        "unit": "percent (paper Table 4 scale)",
        "note": "D.A/M.R/D.E follow the Table-4 three-way partition; label parsing is reconstructed from dataset references.",
    }


def _extract_emotion_label(text: str) -> str | None:
    normalized = _normalize_label(text)

    # Accept direct coarse-label predictions.
    for coarse in sorted(EMOTION_COARSE_TO_POLARITY, key=len, reverse=True):
        if coarse in normalized:
            return coarse

    # Map the exact fine-grained labels listed in Sec. 3.2.2 to coarse labels.
    # Longest first is mostly defensive; these labels do not currently overlap.
    for fine in sorted(EMOTION_FINE_TO_COARSE, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(fine)}(?![a-z])", normalized):
            return EMOTION_FINE_TO_COARSE[fine]
    return None


def emotion_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    strict = relaxed = parsed = 0
    for r in ok:
        ref = _extract_emotion_label(r["reference"])
        pred = _extract_emotion_label(r["prediction"])
        parsed += pred is not None
        if ref is not None and pred is not None:
            strict += pred == ref
            relaxed += EMOTION_COARSE_TO_POLARITY[pred] == EMOTION_COARSE_TO_POLARITY[ref]
    n = len(ok)
    return {
        "n": n,
        "strict_accuracy": 100.0 * strict / n if n else None,
        "relaxed_accuracy": 100.0 * relaxed / n if n else None,
        "parsability_rate_diagnostic": 100.0 * parsed / n if n else None,
        "unit": "percent (paper Table 4 scale)",
        "note": "Fine-to-coarse mapping exactly follows LongSpeech Sec. 3.2.2; relaxed accuracy compares positive/negative/cognitive/neutral polarity.",
    }


def normalized_exact_match_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    correct = sum(normalize_text(r["prediction"]) == normalize_text(r["reference"]) for r in ok)
    n = len(ok)
    return {
        "n": n,
        "normalized_exact_match_proxy": 100.0 * correct / n if n else None,
        "unit": "percent",
    }


def temporal_judge_metrics(judged_records: list[dict]) -> dict:
    total = len(judged_records)
    judge_errors = sum(r.get("judge_error") is not None for r in judged_records)
    usable = [r for r in judged_records if r.get("judge_error") is None]
    counts = Counter(str(r.get("judgment", "")).upper() for r in usable)
    invalid = len(usable) - sum(counts[j] for j in TEMPORAL_JUDGMENTS)
    valid = sum(counts[j] for j in TEMPORAL_JUDGMENTS)

    # Paper scores are defined over judged examples. Surface coverage explicitly
    # so an interrupted/partially-judged run cannot be mistaken for a full-test
    # benchmark result. Invalid judge outputs are not silently counted as NO.
    strict = 100.0 * counts["YES"] / valid if valid else None
    relaxed = 100.0 * (counts["YES"] + counts["PARTIALLY"]) / valid if valid else None
    return {
        "n": valid,
        "judge_rows": total,
        "judge_errors": judge_errors,
        "strict_accuracy": strict,
        "relaxed_accuracy": relaxed,
        "judgment_counts": {j: counts[j] for j in ("YES", "PARTIALLY", "NO")},
        "invalid_judgments": invalid,
        "judge_valid_coverage_pct": 100.0 * valid / total if total else None,
        "unit": "percent (paper Table 4 scale)",
    }


def evaluate_task(task: str, records: list[dict]) -> dict:
    if task == "ASR":
        return asr_metrics(records)
    if task == "translation":
        return translation_metrics(records)
    if task == "summary":
        return summary_metrics(records)
    if task in {"speaker_count", "content_separation"}:
        return numeric_fixed_answer_metrics(records)
    if task == "language_detection":
        return language_detection_metrics(records)
    if task == "emotionQA":
        return emotion_metrics(records)
    if task == "Temporal_Relative_QA":
        result = normalized_exact_match_metrics(records)
        result["note"] = (
            "Diagnostic proxy only. Paper-comparable Temporal metrics require judge_temporal.py, "
            "which applies GPT-4-Turbo YES/PARTIALLY/NO judgments."
        )
        return result
    raise ValueError(f"Unsupported task: {task}")
