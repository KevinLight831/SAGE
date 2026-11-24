#!/usr/bin/env bash
# Ring-A-Bell ASR evaluation against a SAGE text-encoder checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  # shellcheck disable=SC1091
  source scripts/config.env
fi

CUDA_ID="${CUDA_ID:-0}"
EXP_DIR="${EXP_DIR:-./outputs/nudity_sage}"
TARGET_CKPT="${TARGET_CKPT:-${EXP_DIR}/models/TextEncoder-text_encoder_full-epoch_999.pt}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/ringabell}"
ETA="${ETA:-3}"

mkdir -p "${OUT_DIR}"
PYTHONPATH="${ROOT}/generate-scripts${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python attack/ringabell/RingABell.py \
  --csv_file attack/ringabell/data/nudity-ring-a-bell.csv \
  --prompt_file attack/ringabell/data/nudity.csv \
  --img_path "${OUT_DIR}" \
  --output_path "${OUT_DIR}" \
  --eta "${ETA}" \
  --concept nudity \
  --ckpt "${TARGET_CKPT}"
