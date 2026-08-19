#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from longspeech_eval.constants import TASKS, TEST_COUNTS
from longspeech_eval.metrics import evaluate_task, temporal_judge_metrics


def load_records(paths: list[Path], *, id_key: str = "example_id") -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    dedup = {}
    for row in rows:
        dedup[row[id_key]] = row
    return list(dedup.values())


def task_files(result_dir: Path, task: str) -> list[Path]:
    direct = result_dir / f"{task}.jsonl"
    if direct.exists():
        return [direct]
    return sorted(result_dir.glob(f"{task}.shard*-of-*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LongSpeech metrics from saved MiniCPM-o predictions.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument(
        "--temporal-judgments",
        default="results/Temporal_Relative_QA.judged.jsonl",
        help="GPT-4-Turbo judgment JSONL produced by judge_temporal.py",
    )
    args = parser.parse_args()

    tasks = list(TASKS) if args.tasks == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]
    result_dir = Path(args.results_dir)
    metrics = {}

    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"Unknown task: {task}")
        paths = task_files(result_dir, task)
        if not paths:
            print(f"[skip] {task}: no result files")
            continue
        records = load_records(paths)

        if task == "Temporal_Relative_QA" and Path(args.temporal_judgments).exists():
            judged = load_records([Path(args.temporal_judgments)])
            metrics[task] = temporal_judge_metrics(judged)
            metrics[task]["judge_file"] = args.temporal_judgments
            metrics[task]["judge_model"] = next((r.get("judge_model") for r in judged if r.get("judge_model")), None)
            metrics[task]["judge_prompt_version"] = next(
                (r.get("judge_prompt_version") for r in judged if r.get("judge_prompt_version")), None
            )
            metrics[task]["note"] = (
                "Paper protocol: GPT-4-Turbo YES/PARTIALLY/NO. The paper does not publish its exact judge prompt; "
                "this repo records the reconstructed prompt version for reproducibility."
            )
        else:
            metrics[task] = evaluate_task(task, records)

        successful = sum(r.get("error") is None for r in records)
        expected = TEST_COUNTS[task]
        metrics[task]["files"] = [str(p) for p in paths]
        metrics[task]["expected_test_n"] = expected
        metrics[task]["prediction_rows"] = len(records)
        metrics[task]["successful_predictions"] = successful
        metrics[task]["errors"] = len(records) - successful
        metrics[task]["prediction_coverage_pct"] = 100.0 * successful / expected
        metrics[task]["complete_test_set"] = successful == expected

        if task == "Temporal_Relative_QA" and "judge_rows" in metrics[task]:
            judged_valid = metrics[task]["n"]
            metrics[task]["judge_coverage_of_successful_predictions_pct"] = (
                100.0 * judged_valid / successful if successful else None
            )
            metrics[task]["paper_comparable_complete"] = (
                successful == expected
                and judged_valid == successful
                and metrics[task].get("judge_errors", 0) == 0
                and metrics[task].get("invalid_judgments", 0) == 0
            )

        print(f"[{task}] {json.dumps(metrics[task], ensure_ascii=False)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
