#!/usr/bin/env bash
# Train SAGE for Van Gogh style erasure (text_encoder_layer0). If you want to train Monet-style erasure, you need to modify the prompt parameters.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  # shellcheck disable=SC1091
  source scripts/config.env
fi

CUDA_ID="${CUDA_ID:-0}"
EXP_DIR="${EXP_DIR:-./outputs/style_vangogh}"
CKPT_PATH="${CKPT_PATH:-./models/sd-v1-4-full-ema.ckpt}"
DIFFUSERS_UNET_CONFIG="${DIFFUSERS_UNET_CONFIG:-./models/diffusers_unet_config.json}"
TRAIN_METHOD="${TRAIN_METHOD:-text_encoder_layer0}"

mkdir -p "${EXP_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python train-scripts/train_sage.py \
  --attack_init random \
  --attack_step 30 \
  --retain_train reg \
  --warmup_iter 200 \
  --dataset_retain coco_object \
  --prompt vangogh \
  --train_method "${TRAIN_METHOD}" \
  --erase_w 1.0 \
  --retain_image_w 0.5 \
  --retain_text_w 0.5 \
  --attack_lr 1e-3 \
  --retain_batch 32 \
  --retain_batch_img 1 \
  --devices 0,0 \
  --ckpt_path "${CKPT_PATH}" \
  --diffusers_config_path "${DIFFUSERS_UNET_CONFIG}" \
  --result_path "${EXP_DIR}" 2>&1 | tee "${EXP_DIR}/train.log"

echo "Training finished. Next: bash scripts/style/eval.sh"
