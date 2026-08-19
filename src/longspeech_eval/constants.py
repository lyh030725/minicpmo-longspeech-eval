MODEL_ID = "openbmb/MiniCPM-o-4_5"
DATASET_ID = "ATH-MaaS/Marco_Longspeech"

TASKS = (
    "ASR",
    "Temporal_Relative_QA",
    "summary",
    "content_separation",
    "emotionQA",
    "speaker_count",
    "translation",
    "language_detection",
)

TEST_COUNTS = {
    "ASR": 15274,
    "Temporal_Relative_QA": 1262,
    "summary": 937,
    "content_separation": 1263,
    "emotionQA": 1263,
    "speaker_count": 1263,
    "translation": 6309,
    "language_detection": 3170,
}

# MiniCPM-o 4.5 chat() defaults to max_new_tokens=4096. Keep one model-default
# budget for all tasks instead of introducing task-specific generation tuning.
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_SEED = 42
