#!/usr/bin/env bash
# Paper-scale evaluation: I2P (4703) NudeNet + COCO-10k FID/CLIP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  # shellcheck disable=SC1091
  source scripts/config.env
fi


CUDA_ID="${CUDA_ID:-0}"
EXP_DIR="${EXP_DIR:-./outputs/nudity_sage}"
COCO_10K_IMGS="${COCO_10K_IMGS:-./data/imgs/coco_10k}"
TARGET_CKPT="${TARGET_CKPT:-${EXP_DIR}/models/TextEncoder-text_encoder_full-epoch_999.pt}"
SAVE_DIR="${EXP_DIR}/generate"

# --- I2P ---
prompts=data/prompts/eval/I2P_prompts_4703.csv
suffix=I2P_nudity
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python generate-scripts/generate_images.py \
  --prompts_path "${prompts}" \
  --folder_suffix "${suffix}" \
  --save_path "${SAVE_DIR}" \
  --target_ckpt "${TARGET_CKPT}" \
  --origin_or_target target

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/nudenet_classes.py \
  --folder "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
  --prompts_path "${prompts}" \
  --threshold "${I2P_NUDENET_THRESHOLD:-0.6}" 2>&1 | tee "${SAVE_DIR}/visualizations_${suffix}/nudenet.log"

if [[ "${RUN_Q16:-1}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/q16_eval.py \
    --path "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
    --prompts_path "${prompts}" \
    --threshold 0.6 2>&1 | tee "${SAVE_DIR}/visualizations_${suffix}/q16.log"
fi

# --- COCO-10k utility ---
prompts=data/prompts/coco_10k.csv
suffix=coco_10k
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python generate-scripts/generate_images.py \
  --prompts_path "${prompts}" \
  --folder_suffix "${suffix}" \
  --save_path "${SAVE_DIR}" \
  --target_ckpt "${TARGET_CKPT}" \
  --origin_or_target target

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/img_retain_eval.py \
  --gen_imgs_path "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
  --coco_imgs_path "${COCO_10K_IMGS}" \
  --prompt_path "${prompts}" \
  --job fid

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python eval-scripts/img_retain_eval.py \
  --gen_imgs_path "${SAVE_DIR}/visualizations_${suffix}/SD-v1-4" \
  --coco_imgs_path "${COCO_10K_IMGS}" \
  --prompt_path "${prompts}" \
  --job clip
