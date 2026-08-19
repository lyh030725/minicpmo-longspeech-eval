from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel

from .constants import DEFAULT_MAX_NEW_TOKENS, MODEL_ID


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    revision: str = "main"
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


def _torch_dtype(name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MiniCPMOEvaluator:
    def __init__(self, config: ModelConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for MiniCPM-o 4.5 evaluation")
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
            torch_dtype=_torch_dtype(config.dtype),
            init_vision=False,
            init_audio=True,
            init_tts=False,
            low_cpu_mem_usage=True,
        )
        self.model.eval().cuda()

    @torch.inference_mode()
    def generate(self, *, prompt: str, audio: np.ndarray, seed: int) -> str:
        # Keep the dataset's user instruction verbatim and use the upstream native
        # audio content format. No extra system prompt is injected.
        seed_everything(seed)
        msgs = [{"role": "user", "content": [prompt, audio]}]

        # Deliberately do not override do_sample / temperature / top_p / top_k.
        # MiniCPM-o therefore uses its own chat/generation defaults. We only disable
        # hidden thinking and waveform generation because LongSpeech expects text.
        result = self.model.chat(
            msgs=msgs,
            max_new_tokens=self.config.max_new_tokens,
            use_tts_template=False,
            enable_thinking=False,
            generate_audio=False,
        )
        if result is None:
            return ""
        return str(result).strip()
