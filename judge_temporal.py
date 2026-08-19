#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from longspeech_eval.temporal_judge import JUDGE_PROMPT_VERSION, JudgeConfig, judge_one


def load_predictions(result_dir: Path) -> list[dict]:
    direct = result_dir / "Temporal_Relative_QA.jsonl"
    paths = [direct] if direct.exists() else sorted(result_dir.glob("Temporal_Relative_QA.shard*-of-*.jsonl"))
    if not paths:
        raise FileNotFoundError("No Temporal_Relative_QA prediction file(s) found")

    dedup: dict[str, dict] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    dedup[row["example_id"]] = row
    return sorted(dedup.values(), key=lambda r: int(r.get("index", 0)))


def completed_ids(path: Path, *, model: str, prompt_version: str) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("judge_error") is None
                and row.get("judgment") in {"YES", "PARTIALLY", "NO"}
                and row.get("judge_model") == model
                and row.get("judge_prompt_version") == prompt_version
            ):
                done.add(row["example_id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge LongSpeech Temporal QA with GPT-4-Turbo as described in the paper.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/Temporal_Relative_QA.judged.jsonl")
    parser.add_argument("--model", default="gpt-4-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-s", type=float, default=2.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for GPT-4-Turbo temporal judging")

    rows = [r for r in load_predictions(Path(args.results_dir)) if r.get("error") is None]
    rows = rows[args.start:]
    if args.limit is not None:
        rows = rows[: args.limit]

    config = JudgeConfig(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output, model=config.model, prompt_version=JUDGE_PROMPT_VERSION) if args.resume else set()
    client = OpenAI()

    with output.open("a", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Temporal GPT-4-Turbo judge", unit="sample"):
            if row["example_id"] in done:
                continue

            judged = {
                "example_id": row["example_id"],
                "index": row.get("index"),
                "question": row.get("prompt", ""),
                "reference": row.get("reference", ""),
                "prediction": row.get("prediction", ""),
                "judgment": None,
                "judge_raw": None,
                "judge_model": config.model,
                "judge_resolved_model": None,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "judge_temperature": config.temperature,
                "judge_error": None,
            }

            for attempt in range(args.max_retries + 1):
                try:
                    judgment, raw, resolved_model = judge_one(
                        client,
                        config=config,
                        question=judged["question"],
                        reference=judged["reference"],
                        prediction=judged["prediction"],
                    )
                    judged["judge_raw"] = raw
                    judged["judge_resolved_model"] = resolved_model
                    if judgment is None:
                        raise ValueError(f"Unparseable judge output: {raw!r}")
                    judged["judgment"] = judgment
                    break
                except Exception as exc:
                    judged["judge_error"] = f"{type(exc).__name__}: {exc}"
                    if attempt >= args.max_retries:
                        break
                    time.sleep(args.retry_base_s * (2**attempt))

            if judged["judgment"] is not None:
                judged["judge_error"] = None
            fout.write(json.dumps(judged, ensure_ascii=False) + "\n")
            fout.flush()

            if args.fail_fast and judged["judge_error"] is not None:
                raise RuntimeError(judged["judge_error"])


if __name__ == "__main__":
    main()
