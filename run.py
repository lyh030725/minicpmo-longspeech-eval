#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import repo_info
from tqdm import tqdm

from longspeech_eval.audio import load_audio_16k_mono, resolve_audio_path
from longspeech_eval.constants import DATASET_ID, DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, MODEL_ID, TASKS
from longspeech_eval.dataset import iter_examples, metadata_file
from longspeech_eval.model import MiniCPMOEvaluator, ModelConfig


def completed_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("error") is None and row.get("example_id"):
                    done.add(row["example_id"])
            except json.JSONDecodeError:
                continue
    return done


def write_environment(output_dir: Path, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_sha = None
    dataset_sha = None
    try:
        model_sha = repo_info(args.model_id, repo_type="model", revision=args.model_revision).sha
    except Exception:
        pass
    try:
        dataset_sha = repo_info(args.dataset_id, repo_type="dataset", revision=args.dataset_revision).sha
    except Exception:
        pass

    info = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_resolved_sha": model_sha,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "dataset_resolved_sha": dataset_sha,
        "seed": args.seed,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "use_tts_template": False,
            "enable_thinking": False,
            "generate_audio": False,
            "sampling": "MiniCPM-o chat defaults (do_sample/temperature/top_p/top_k not overridden)",
            "system_prompt": None,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "environment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def example_seed(base_seed: int, task_index: int, row_index: int) -> int:
    # Stable across resume/sharding without depending on Python's salted hash().
    return int(base_seed + task_index * 1_000_003 + row_index)


def run_task(args, evaluator: MiniCPMOEvaluator, task: str, task_index: int) -> None:
    meta_path = metadata_file(args.metadata_dir, task)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run python scripts/download_metadata.py first")

    suffix = f".shard{args.shard_index:02d}-of-{args.num_shards:02d}" if args.num_shards > 1 else ""
    out_path = Path(args.output_dir) / f"{task}{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(out_path) if args.resume else set()

    iterator = iter_examples(
        meta_path,
        task=task,
        start=args.start,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    with out_path.open("a", encoding="utf-8") as fout:
        for ex in tqdm(iterator, desc=task, unit="sample"):
            if ex.example_id in done:
                continue

            seed = example_seed(args.seed, task_index, ex.index)
            record = {
                "example_id": ex.example_id,
                "index": ex.index,
                "task": ex.task,
                "language": ex.language,
                "audio_relpath": ex.audio_relpath,
                "prompt": ex.prompt,
                "reference": ex.reference,
                "prediction": None,
                "audio_duration_s": None,
                "latency_s": None,
                "seed": seed,
                "error": None,
            }

            audio = None
            try:
                audio_path = resolve_audio_path(
                    ex.audio_relpath,
                    audio_root=args.audio_root,
                    cache_dir=args.cache_dir,
                    dataset_id=args.dataset_id,
                    revision=args.dataset_revision,
                )
                audio, duration_s = load_audio_16k_mono(audio_path)
                record["audio_duration_s"] = duration_s

                if args.max_audio_seconds is not None and duration_s > args.max_audio_seconds:
                    raise RuntimeError(
                        f"audio_duration_s={duration_s:.2f} exceeds --max-audio-seconds={args.max_audio_seconds}"
                    )

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prediction = evaluator.generate(prompt=ex.prompt, audio=audio, seed=seed)
                torch.cuda.synchronize()
                record["latency_s"] = time.perf_counter() - t0
                record["prediction"] = prediction
            except torch.cuda.OutOfMemoryError as exc:
                record["error"] = f"CUDA_OOM: {exc}"
                torch.cuda.empty_cache()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if audio is not None:
                    del audio

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if args.fail_fast and record["error"] is not None:
                raise RuntimeError(record["error"])


def parse_tasks(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    tasks = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in tasks if x not in TASKS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown task(s): {unknown}; valid={list(TASKS)}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniCPM-o 4.5 on Marco-LongSpeech test examples.")
    parser.add_argument("--tasks", default="all", help="all or comma-separated task names")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--audio-root", default=None, help="Local dataset root; otherwise audio downloads on demand from HF")
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME"))
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-audio-seconds", type=float, default=None, help="Optional safety cap; unset = native full audio")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")

    output_dir = Path(args.output_dir)
    write_environment(output_dir, args)

    config = ModelConfig(
        model_id=args.model_id,
        revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    evaluator = MiniCPMOEvaluator(config)

    task_to_index = {task: i for i, task in enumerate(TASKS)}
    for task in tasks:
        run_task(args, evaluator, task, task_to_index[task])


if __name__ == "__main__":
    main()
