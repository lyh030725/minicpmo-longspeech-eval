from __future__ import annotations

import re
from dataclasses import dataclass

from .metrics import TEMPORAL_JUDGMENTS


# The paper specifies GPT-4-Turbo and the three labels, but does not publish the
# exact judge prompt. Keep this reconstructed prompt versioned and saved with each
# judgment so the evaluation itself is reproducible.
JUDGE_PROMPT_VERSION = "longspeech-reconstructed-v1"

SYSTEM_PROMPT = """You are an evaluator for the LongSpeech Temporal Issue Localization benchmark.
Judge whether a candidate answer correctly answers the question when compared with the reference answer.
Return exactly one label and nothing else:
YES = fully correct
PARTIALLY = partially correct, incomplete, or contains the correct core information with a meaningful omission/addition
NO = incorrect, contradictory, irrelevant, or does not answer the question
Judge semantic correctness rather than exact wording."""


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "gpt-4-turbo"
    temperature: float = 0.0
    max_tokens: int = 8


def make_user_prompt(*, question: str, reference: str, prediction: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Reference answer:\n"
        f"{reference}\n\n"
        "Candidate answer:\n"
        f"{prediction}\n\n"
        "Judgment:"
    )


def parse_judgment(text: str) -> str | None:
    cleaned = str(text).strip().upper()
    if cleaned in TEMPORAL_JUDGMENTS:
        return cleaned
    # Robustness to accidental punctuation/one-line explanation.
    match = re.match(r"^(PARTIALLY|YES|NO)\b", cleaned)
    return match.group(1) if match else None


def judge_one(
    client, *, config: JudgeConfig, question: str, reference: str, prediction: str
) -> tuple[str | None, str, str | None]:
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(question=question, reference=reference, prediction=prediction)},
        ],
    )
    raw = response.choices[0].message.content or ""
    return parse_judgment(raw), raw, getattr(response, "model", None)
