# Model files (not shipped in git)

Place the following files under `models/` before training:

## Stable Diffusion v1.4 (CompVis)

Download [`sd-v1-4-full-ema.ckpt`](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and save as:

```text
models/sd-v1-4-full-ema.ckpt
```

`models/diffusers_unet_config.json` is already included (used when converting / saving Diffusers-format weights).

## Released SAGE checkpoints

Put paper checkpoints (nudity / style) under e.g. `outputs/` or any path, then point scripts via:

```bash
export TARGET_CKPT=/path/to/TextEncoder-....pt
```

## Style classifier

`scripts/style/eval.sh` needs a HuggingFace-style image classifier checkpoint directory:

```text
models/style_classifier/checkpoint-2800/
```

Set `STYLE_CLASSIFIER_DIR` in `scripts/config.env`.

## COCO images for FID / CLIP

- Quick loop: `data/imgs/coco_200` matching `data/prompts/coco_200.csv`
- Paper utility: `data/imgs/coco_10k` matching `data/prompts/coco_10k.csv`
