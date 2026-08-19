from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .constants import TASKS


@dataclass(frozen=True)
class Example:
    index: int
    task: str
    language: str
    audio_relpath: str
    prompt: str
    reference: str

    @property
    def example_id(self) -> str:
        return f"{self.task}:{self.index:06d}"


def metadata_file(metadata_dir: str | Path, task: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return Path(metadata_dir) / f"{task}.test.jsonl"


def _coerce_messages(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"messages must be a list, got {type(value).__name__}")
    result = []
    for msg in value:
        if isinstance(msg, str):
            msg = json.loads(msg)
        if not isinstance(msg, dict):
            raise ValueError("Each message must be a dict")
        result.append(msg)
    return result


def parse_row(row: dict, index: int, expected_task: str | None = None) -> Example:
    task = row.get("task")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"Expected task={expected_task!r}, found {task!r} at row {index}")

    messages = _coerce_messages(row["messages"])
    if len(messages) < 2:
        raise ValueError(f"Expected user + assistant messages at row {index}")

    user = messages[0]
    assistant = messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"Unexpected message roles at row {index}")

    audio_relpath = user.get("audio")
    prompt = user.get("content")
    reference = assistant.get("content")
    if not isinstance(audio_relpath, str) or not audio_relpath:
        raise ValueError(f"Missing audio path at row {index}")
    if not isinstance(prompt, str):
        raise ValueError(f"Missing prompt at row {index}")
    if not isinstance(reference, str):
        raise ValueError(f"Missing reference at row {index}")

    return Example(
        index=index,
        task=str(task),
        language=str(row.get("language", "")),
        audio_relpath=audio_relpath,
        prompt=prompt,
        reference=reference,
    )


def iter_examples(
    path: str | Path,
    *,
    task: str,
    start: int = 0,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[Example]:
    if start < 0:
        raise ValueError("start must be >= 0")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    yielded = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < start:
                continue
            if (index - start) % num_shards != shard_index:
                continue
            if limit is not None and yielded >= limit:
                break
            row = json.loads(line)
            yield parse_row(row, index=index, expected_task=task)
            yielded += 1
