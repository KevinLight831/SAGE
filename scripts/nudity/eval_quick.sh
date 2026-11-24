#!/usr/bin/env bash
# Small-scale nudity / utility checks for fast iteration
# (filtered nudity prompts + 142 nudity + coco_200).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  # shellcheck disable=SC1091
  source scripts/config.env
fi

CUDA_ID="${CUDA_ID:-0}"
EXP_DIR="${EXP_DIR:-./outputs/nudity_sage}"
COCO_200_IMGS="${COCO_200_IMGS:-./data/imgs/coco_200}"
TARGET_CKPT="${TARGET_CKPT:-${EXP_DIR}/models/TextEncoder-text_encoder_full-epoch_999.pt}"
SAVE_DIR="${EXP_DIR}/generate"

run_gen_eval () {
  local prompts="$1"
  local suffix="$2"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" python generate-scripts/generate_images.py \
    --prompts_path "${prompts}" \
    --folder_suffix "${suffix}" \
    --save_path "${SAVE_DIR}" \
    --target_ckpt "${TARGET_CKPT}" \
    --origin_or_target target

  python eval-scripts/nudenet_classes.py \
    --folder "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
    --prompts_path "${prompts}" \
    --threshold "${QUICK_NUDENET_THRESHOLD:-0.5}" 2>&1 | tee "${SAVE_DIR}/visualizations_${suffix}/nudenet.log"
}

run_gen_eval data/prompts/eval/nudity_quick.csv nudity_quick
run_gen_eval data/prompts/eval/nudity.csv nudity_142

# Utility on coco_200
suffix=coco_200
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python generate-scripts/generate_images.py \
  --prompts_path data/prompts/coco_200.csv \
  --folder_suffix "${suffix}" \
  --save_path "${SAVE_DIR}" \
  --target_ckpt "${TARGET_CKPT}" \
  --origin_or_target target

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/img_retain_eval.py \
  --gen_imgs_path "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
  --coco_imgs_path "${COCO_200_IMGS}" \
  --prompt_path data/prompts/coco_200.csv \
  --job fid

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/img_retain_eval.py \
  --gen_imgs_path "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
  --coco_imgs_path "${COCO_200_IMGS}" \
  --prompt_path data/prompts/coco_200.csv \
  --job clip


