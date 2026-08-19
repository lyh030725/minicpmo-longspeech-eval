#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys


MIN_RECOMMENDED_VRAM_GIB = 28.0
SAFER_LONGSPEECH_VRAM_GIB = 40.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-install", action="store_true")
    args = parser.parse_args()

    import torch

    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gib = props.total_memory / (1024**3)
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"gpu_total_vram_gib={total_vram_gib:.2f}")
        if total_vram_gib < MIN_RECOMMENDED_VRAM_GIB:
            print(
                "WARNING: GPU VRAM is below MiniCPM-o 4.5's official >=28 GB "
                "PyTorch deployment guidance. Native BF16 full-audio LongSpeech "
                "evaluation is expected to OOM on many ~10-minute samples."
            )
        elif total_vram_gib < SAFER_LONGSPEECH_VRAM_GIB:
            print(
                "NOTE: GPU meets the official >=28 GB guidance, but LongSpeech "
                "uses ~10-minute audio. 40/48 GB+ provides safer activation headroom."
            )

    if not str(torch.__version__).startswith("2.9.1"):
        raise SystemExit(f"Expected RunPod image torch 2.9.1, got {torch.__version__}")

    if not args.pre_install:
        import transformers
        import huggingface_hub
        import soundfile

        print(f"transformers={transformers.__version__}")
        print(f"huggingface_hub={huggingface_hub.__version__}")
        print(f"soundfile={soundfile.__version__}")
        if transformers.__version__ != "4.51.0":
            raise SystemExit(f"Expected transformers 4.51.0, got {transformers.__version__}")


if __name__ == "__main__":
    main()
