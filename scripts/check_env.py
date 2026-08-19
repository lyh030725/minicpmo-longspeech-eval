#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys


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
        print(f"gpu={torch.cuda.get_device_name(0)}")

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
