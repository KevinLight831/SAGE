#!/usr/bin/env bash
# Style erase (vangogh / monet) + style retain (other artists) evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  # shellcheck disable=SC1091
  source scripts/config.env
fi

CUDA_ID="${CUDA_ID:-0}"
STYLE="${STYLE:-vangogh}"   # vangogh | monet
EXP_DIR="${EXP_DIR:-./outputs/style_${STYLE}}"
STYLE_CLASSIFIER_DIR="${STYLE_CLASSIFIER_DIR:-./models/style_classifier/checkpoint-2800}"
TRAIN_METHOD="${TRAIN_METHOD:-text_encoder_layer0}"
TARGET_CKPT="${TARGET_CKPT:-${EXP_DIR}/models/TextEncoder-${TRAIN_METHOD}-epoch_999.pt}"
SAVE_DIR="${EXP_DIR}/generate"

if [[ "${STYLE}" == "monet" ]]; then
  ERASE_PROMPTS=data/prompts/style/monet.csv
  RETAIN_PROMPTS=data/prompts/style/monet_other_artists.csv
  ERASE_ARTIST=claude-monet
  ERASE_SUFFIX=monet
  RETAIN_SUFFIX=monet_other_artists
elif [[ "${STYLE}" == "vangogh" ]]; then
  ERASE_PROMPTS=data/prompts/eval/vangogh.csv
  RETAIN_PROMPTS=data/prompts/eval/other_artists.csv
  ERASE_ARTIST=vincent-van-gogh
  ERASE_SUFFIX=vangogh
  RETAIN_SUFFIX=other_artists
else
  echo "STYLE must be vangogh or monet" >&2
  exit 2
fi

ERASE_EDITED_PATH="${ERASE_EDITED_PATH:-}"
RETAIN_EDITED_PATH="${RETAIN_EDITED_PATH:-}"
mkdir -p "${EXP_DIR}"
if [[ -z "${ERASE_EDITED_PATH}" && -z "${RETAIN_EDITED_PATH}" ]]; then
  for split in erase retain; do
    if [[ "${split}" == erase ]]; then
      prompts="${ERASE_PROMPTS}"
      suffix="${ERASE_SUFFIX}"
    else
      prompts="${RETAIN_PROMPTS}"
      suffix="${RETAIN_SUFFIX}"
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python generate-scripts/generate_images.py \
      --prompts_path "${prompts}" --folder_suffix "${suffix}" --save_path "${SAVE_DIR}" \
      --target_ckpt "${TARGET_CKPT}" --origin_or_target target
    if [[ "${split}" == erase ]]; then
      ERASE_EDITED_PATH="${SAVE_DIR}/visualizations_${suffix}/SD-v1-4"
    else
      RETAIN_EDITED_PATH="${SAVE_DIR}/visualizations_${suffix}/SD-v1-4"
    fi
  done
elif [[ -z "${ERASE_EDITED_PATH}" || -z "${RETAIN_EDITED_PATH}" ]]; then
  echo "Set both ERASE_EDITED_PATH and RETAIN_EDITED_PATH" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/style_eval.py \
  --erase_folder "${ERASE_EDITED_PATH}" \
  --retain_folder "${RETAIN_EDITED_PATH}" \
  --prompts_path "${ERASE_PROMPTS}" \
  --retain_prompts_path "${RETAIN_PROMPTS}" \
  --classifier_dir "${STYLE_CLASSIFIER_DIR}" \
  --erase_artist "${ERASE_ARTIST}" 2>&1 | tee "${EXP_DIR}/style_score.log"

# Optional LPIPS erase/retain if you also have original SD generations
if [[ -n "${ERASE_ORIGINAL_PATH:-}" && -n "${RETAIN_ORIGINAL_PATH:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/lpips_class.py \
    --erase_prompts_path "${ERASE_PROMPTS}" \
    --retain_prompts_path "${RETAIN_PROMPTS}" \
    --erase_original_path "${ERASE_ORIGINAL_PATH}" \
    --retain_original_path "${RETAIN_ORIGINAL_PATH}" \
    --erase_edited_path "${ERASE_EDITED_PATH}" \
    --retain_edited_path "${RETAIN_EDITED_PATH}" \
    2>&1 | tee "${EXP_DIR}/lpips_score.log"
fi
