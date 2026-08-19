#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, repo_info

from longspeech_eval.constants import DATASET_ID, TASKS, TEST_COUNTS


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Marco-LongSpeech test JSONL metadata only.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = repo_info(args.dataset_id, repo_type="dataset", revision=args.revision)
    print(f"dataset={args.dataset_id} revision={args.revision} resolved_sha={info.sha}")

    for task in TASKS:
        remote = f"LongSpeechQA/{task}/test.jsonl"
        cached = Path(hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=remote,
            revision=args.revision,
            cache_dir=args.cache_dir,
        ))
        dest = out_dir / f"{task}.test.jsonl"
        shutil.copy2(cached, dest)
        actual = count_lines(dest)
        expected = TEST_COUNTS[task]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{task:24s} rows={actual:6d} expected={expected:6d} [{status}]")


if __name__ == "__main__":
    main()
