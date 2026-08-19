#!/usr/bin/env bash
set -euo pipefail

python scripts/download_metadata.py
python run.py --tasks ASR --limit 1 --fail-fast
python evaluate.py --tasks ASR
