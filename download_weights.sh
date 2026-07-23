#!/usr/bin/env bash
# Downloads the LongCat-Video model weights from Hugging Face into ./weights.
# Requires enough local disk space for all three checkpoints.
set -euo pipefail

pip install "huggingface_hub[cli]"

huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir ./weights/LongCat-Video-Avatar-1.5
