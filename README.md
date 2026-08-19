# MiniCPM-o 4.5 × Marco-LongSpeech

Evaluation harness for running **`openbmb/MiniCPM-o-4_5`** on the **Marco-LongSpeech test set** (`ATH-MaaS/Marco_Longspeech`) on RunPod.

Target RunPod image:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

The public test split contains **30,741 task examples across 8 tasks**. Audio is approximately long-form (~10 minutes per segment), and the same audio can be reused by multiple tasks.

## Reproduction policy

The inference path is intentionally conservative:

- use each dataset row's **original user prompt verbatim**;
- add **no custom system prompt**;
- pass the **full audio natively** (no segmentation/chunking by default);
- use MiniCPM-o's offline `model.chat()` audio-understanding path;
- use the official MiniCPM-o audio-understanding input order `[task_prompt, audio_ndarray]`;
- set `use_tts_template=False`, `enable_thinking=False`, and `generate_audio=False` because LongSpeech evaluates text output, not speech generation;
- do **not** override `do_sample`, `temperature`, `top_p`, or `top_k`, so MiniCPM-o keeps its own generation defaults;
- keep MiniCPM-o's `chat()` default output budget, `max_new_tokens=4096`;
- leave upstream input-length/slicing limits at their `chat()` defaults rather than adding benchmark-specific truncation logic;
- use a stable per-example seed derived from base seed `42`, making resume/sharding independent of evaluation order;
- save raw predictions even when a paper-exact metric implementation is not publicly available.

## RunPod quick start

Create a pod with:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

Then:

```bash
git clone https://github.com/lyh030725/minicpmo-longspeech-eval.git
cd minicpmo-longspeech-eval

bash scripts/setup_runpod.sh

export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/download_metadata.py
```

### Smoke test

Start with one sample:

```bash
python run.py --tasks ASR --limit 1 --fail-fast
```

Then 10 samples:

```bash
python run.py --tasks ASR --limit 10
python evaluate.py --tasks ASR
```

### Full test set

```bash
python run.py --tasks all
python evaluate.py --tasks all
```

The runner resumes successful examples automatically.

For a one-command smoke test after setup:

```bash
bash scripts/smoke_test.sh
```

## Data download behavior

`download_metadata.py` downloads only the 8 small `LongSpeechQA/<task>/test.jsonl` metadata files.

During inference, each WAV is downloaded **on demand** with `hf_hub_download()` and retained in the Hugging Face cache. This avoids downloading the full ~2 TB repository up front and naturally reuses cached audio when different tasks reference the same WAV. The complete test run can still consume a large amount of persistent storage, so a RunPod Network Volume or sufficiently large `/workspace` volume is recommended.

If the audio tree has already been downloaded locally, bypass Hugging Face audio downloads with:

```bash
python run.py --tasks ASR --audio-root /workspace/Marco_Longspeech
```

The directory must contain paths such as:

```text
/workspace/Marco_Longspeech/LongSpeech_p1/wavs/...
/workspace/Marco_Longspeech/LongSpeech_p2/wavs/...
/workspace/Marco_Longspeech/LongSpeech_p3/wavs/...
```

## Multi-GPU / multi-pod sharding

Run independent shards into separate result files:

```bash
# GPU/pod 0
python run.py --tasks ASR --num-shards 4 --shard-index 0

# GPU/pod 1
python run.py --tasks ASR --num-shards 4 --shard-index 1

# GPU/pod 2
python run.py --tasks ASR --num-shards 4 --shard-index 2

# GPU/pod 3
python run.py --tasks ASR --num-shards 4 --shard-index 3
```

`evaluate.py` automatically discovers `ASR.shardXX-of-YY.jsonl` files when `ASR.jsonl` is absent.

## Result format

Each line in `results/<task>.jsonl` contains:

```json
{
  "example_id": "ASR:000000",
  "index": 0,
  "task": "ASR",
  "language": "en",
  "audio_relpath": "LongSpeech_p3/wavs/part_00/005259.wav",
  "prompt": "Detect the language and recognize the speech: <|en|>",
  "reference": "...",
  "prediction": "...",
  "audio_duration_s": 600.1,
  "latency_s": 31.2,
  "seed": 42,
  "error": null
}
```

`results/environment.json` records model/dataset revision SHAs and runtime configuration.

`evaluate.py` also records `expected_test_n`, `successful_predictions`, `prediction_coverage_pct`, and `complete_test_set` for every task. This prevents an interrupted run from silently producing a deceptively high score on only the successful subset. Temporal QA additionally reports judge coverage and `paper_comparable_complete`, which is true only when the complete public test set has successful MiniCPM-o predictions and valid judge labels.

## Metrics

`evaluate.py` follows the public LongSpeech paper as closely as the released material allows. The paper defines the metric protocol, but does **not** release the evaluator source or the exact GPT-4-Turbo judge prompt.

| Task | Metric in this repo | Reproduction status |
|---|---|---|
| ASR | Non-CJK WER, CJK CER, overall CER | paper metric; CJK grouped by dataset language; normalization reconstructed |
| translation | case-insensitive BLEU-4 | paper metric |
| summary | ROUGE-1/2/L F1 (0-100) | paper metric |
| speaker_count | Numeric Accuracy, Parsability Rate, Post-Parsing Precision, Misunderstanding Rate | paper formulas; numeric parser reconstructed |
| content_separation | same four fixed-answer metrics | paper formulas; numeric parser reconstructed |
| language_detection | Detection Accuracy, Misunderstanding Rate, Detection Errors | Table-4 three-way partition reproduced |
| emotionQA | 7-category Strict Accuracy + polarity Relaxed Accuracy | paper mapping reproduced exactly |
| Temporal_Relative_QA | GPT-4-Turbo YES/PARTIALLY/NO -> Strict/Relaxed Accuracy | paper protocol; judge prompt reconstructed |

### ASR CJK grouping

The paper defines CJK as **Chinese, Japanese, and Korean**. The evaluator therefore uses each dataset row's `language` field to decide whether it belongs to the CJK CER group; it does not guess the group from Unicode characters in the transcript. A transcript-character fallback is used only for external/legacy result files that lack the language field.

### Emotion mapping: paper-exact

Section 3.2.2 maps fine-grained predictions to these seven coarse categories. The evaluator implements this mapping literally:

```text
Positive-HighArousal: Excited, Amused
Positive-Uplifting: Hopeful, Determined, Inspirational, Awe, Vivid
Negative-Sadness/Reflection: Melancholy, Somber, Pensive, Reflective
Negative-Tension/Seriousness: Suspenseful, Serious, Dramatic, Urgent, Intense
Cognitive-Curiosity: Intrigued, Questioning
Cognitive-Uncertainty: Confused
Neutral-Informative: Neutral, Calm, Objective, Informative, Assertive
```

Strict Accuracy compares the mapped seven-way category. Relaxed Accuracy compares only the broader polarity: `positive`, `negative`, `cognitive`, or `neutral`. As in Table 4, these scores are emitted on a **0-100 percentage scale**.

### Language Detection: Table-4 format

The paper reports three columns that partition the examples:

```text
D.A = Detection Accuracy
M.R = Misunderstanding Rate
D.E = Detection Errors
```

This repo reports the same names on the paper's **0-100 percentage scale**. A recognized correct language contributes to D.A; a recognized but wrong language contributes to M.R; an answer from which no language label can be parsed contributes to D.E. The parser learns all ground-truth language spellings from the benchmark result set and also supports common names/codes.

### Fixed-answer tasks

For Speaker Count and Content Separation, Section 3.2.1 defines:

```text
Numeric Accuracy = correct / all
Parsability Rate = parsed / all
Post-Parsing Precision = correct / parsed
Misunderstanding Rate = unparsed / all
```

The relations above are also exactly consistent with the values in the paper's Table 4 (`Pr = N.A / P.R`, `M.R = 100 - P.R` when expressed as percentages). These metrics are emitted on the paper's **0-100 scale**. The authors do not publish the parser implementation, so this repo reconstructs parsing by extracting a numeric answer from the model response. Raw predictions are always retained.

### Temporal Issue Localization: GPT-4-Turbo judge

The paper evaluates Temporal Issue Localization with **GPT-4-Turbo** and three judgments:

```text
YES       = fully correct
PARTIALLY = partially correct
NO        = incorrect

Strict Accuracy  = 100 * YES / N
Relaxed Accuracy = 100 * (YES + PARTIALLY) / N
```

Run MiniCPM-o inference first:

```bash
python run.py --tasks Temporal_Relative_QA
```

Then export an OpenAI API key and run the judge:

```bash
export OPENAI_API_KEY=...
python judge_temporal.py
python evaluate.py --tasks Temporal_Relative_QA
```

The default judge model is exactly the family named by the paper:

```text
gpt-4-turbo
```

You can override it with `--model`, but doing so is no longer the paper's stated judge setup. The current OpenAI API still exposes `gpt-4-turbo`. Each judgment records the requested model, the model identifier returned by the API, temperature, raw judge output, and judge-prompt version.

**Important limitation:** the LongSpeech paper states the judge model and the `YES/PARTIALLY/NO` protocol but does not publish the exact GPT-4-Turbo prompt. Therefore `longspeech-reconstructed-v1` is a minimal reconstructed judge prompt. This is the only unavoidable prompt-level difference unless the authors release their evaluator. The prompt is versioned in every output row so results remain reproducible.

If no judged JSONL exists, `evaluate.py` reports normalized exact match only as a clearly named diagnostic proxy; do not compare that proxy with the paper's Temporal scores.

### What is and is not changed from LongSpeech

For the intended experiment, the benchmark side stays fixed:

```text
LongSpeech public test split
+ original audio
+ original task prompt
+ original reference answer
+ full native long-audio input
+ paper metric definitions
                 ↓
        MiniCPM-o 4.5 only
```

The remaining non-bit-exact pieces are evaluator internals that the authors did not publish: ASR text normalization, the fixed-answer parser, and the exact GPT-4-Turbo judge prompt. They are explicitly marked and versioned rather than silently treated as official code.

## Why torch 2.9.1 is preserved

The MiniCPM-o model card currently recommends `transformers==4.51.0` and documents a tested PyTorch range up to 2.8.0. The requested RunPod image, however, ships torch/torchaudio 2.9.1. This repo intentionally preserves that image stack via `constraints-runpod.txt`, matching the environment pattern already used for MiniCPM-o 4.5 experiments on this RunPod image.

If the upstream model hits a torch-2.9-specific incompatibility, record the exact error before changing the PyTorch stack; changing torch changes the requested reproduction environment.

## Useful commands

```bash
# One specific task
python run.py --tasks summary

# Several tasks
python run.py --tasks speaker_count,language_detection,emotionQA

# Start at row 100 and evaluate 20 selected shard examples
python run.py --tasks ASR --start 100 --limit 20

# Disable resume
python run.py --tasks ASR --limit 10 --no-resume

# Native full audio is the default. Optional debugging cap only:
python run.py --tasks ASR --limit 1 --max-audio-seconds 700

# Run unit tests
pytest -q
```

## Sources

- MiniCPM-o 4.5 model: `openbmb/MiniCPM-o-4_5`
- Marco-LongSpeech dataset: `ATH-MaaS/Marco_Longspeech`
- LongSpeech paper: arXiv `2601.13539`
