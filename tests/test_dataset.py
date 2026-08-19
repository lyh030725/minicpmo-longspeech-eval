import json

from longspeech_eval.dataset import iter_examples, parse_row


def row(task="ASR", audio="LongSpeech_p1/wavs/a.wav", prompt="transcribe", ref="hello"):
    return {
        "language": "en",
        "task": task,
        "messages": [
            {"role": "user", "audio": audio, "content": prompt},
            {"role": "assistant", "content": ref},
        ],
    }


def test_parse_row():
    ex = parse_row(row(), 7, expected_task="ASR")
    assert ex.example_id == "ASR:000007"
    assert ex.audio_relpath.endswith("a.wav")
    assert ex.reference == "hello"


def test_iter_shards(tmp_path):
    p = tmp_path / "ASR.test.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for _ in range(6):
            f.write(json.dumps(row()) + "\n")
    a = list(iter_examples(p, task="ASR", shard_index=0, num_shards=2))
    b = list(iter_examples(p, task="ASR", shard_index=1, num_shards=2))
    assert [x.index for x in a] == [0, 2, 4]
    assert [x.index for x in b] == [1, 3, 5]
