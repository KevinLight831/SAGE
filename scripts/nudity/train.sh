#!/usr/bin/env bash
# Train SAGE on nudity erasure (paper default hyper-params from my_train_2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f scripts/config.env ]]; then
  source scripts/config.env
fi

CUDA_ID="${CUDA_ID:-0}"
EXP_DIR="${EXP_DIR:-./outputs/nudity_sage}"
CKPT_PATH="${CKPT_PATH:-./models/sd-v1-4-full-ema.ckpt}"
DIFFUSERS_UNET_CONFIG="${DIFFUSERS_UNET_CONFIG:-./models/diffusers_unet_config.json}"

mkdir -p "${EXP_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python train-scripts/train_sage.py \
  --attack_init random \
  --attack_step 30 \
  --retain_train reg \
  --warmup_iter 200 \
  --dataset_retain coco_object \
  --prompt nudity \
  --train_method text_encoder_full \
  --erase_w 1.0 \
  --retain_image_w 1.0 \
  --retain_text_w 0.1 \
  --attack_lr 1e-3 \
  --retain_batch 32 \
  --retain_batch_img 4 \
  --negative_guidance 1 \
  --devices 0,0 \
  --ckpt_path "${CKPT_PATH}" \
  --diffusers_config_path "${DIFFUSERS_UNET_CONFIG}" \
  --result_path "${EXP_DIR}" 2>&1 | tee "${EXP_DIR}/train.log"

echo "Training finished. Checkpoint under ${EXP_DIR}/models/"

