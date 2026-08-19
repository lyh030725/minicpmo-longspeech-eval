from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

from .constants import DATASET_ID

TARGET_SR = 16000


def resolve_audio_path(
    audio_relpath: str,
    *,
    audio_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    dataset_id: str = DATASET_ID,
    revision: str = "main",
) -> Path:
    if audio_root is not None:
        local_path = Path(audio_root) / audio_relpath
        if local_path.exists():
            return local_path
        raise FileNotFoundError(f"Audio not found below --audio-root: {local_path}")

    path = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename=audio_relpath,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(path)


def load_audio_16k_mono(path: str | Path) -> tuple[np.ndarray, float]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"Expected mono/stereo waveform, got shape={wav.shape}")

    if int(sr) != TARGET_SR:
        gcd = math.gcd(int(sr), TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // gcd, int(sr) // gcd).astype(np.float32, copy=False)
        sr = TARGET_SR

    duration_s = float(len(wav)) / float(sr)
    return np.ascontiguousarray(wav, dtype=np.float32), duration_s
