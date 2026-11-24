# SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing

[arXiv](https://arxiv.org/abs/2506.09363)  | [Hugging Face](https://huggingface.co/kevin1103/SAGE)

SAGE is a robust semantic-augmentation framework for erasing unsafe concepts and artistic styles from Stable Diffusion v1.4 while retaining general image quality. This release includes training, generation, evaluation, style experiments, cross-backbone generation, and Ring-A-Bell evaluation.

<div align="center">
<img src="assets/method.png" width="90%">
<img src="assets/nudity_table.png" width="90%">
<img src="assets/nudity_fig.png" width="90%">
<img src="assets/style_table.png" width="90%">
<img src="assets/style_fig.png" width="90%">
</div>

## Layout

- `train-scripts/` contains SAGE training, attack, loss, prompt, and checkpoint utilities.
- `generate-scripts/` contains Stable Diffusion, SLD, MACE, and cross-backbone generation tools.
- `eval-scripts/` contains NudeNet, Q16, FID, CLIP, style, and LPIPS metrics.
- `scripts/nudity/` contains quick and paper-scale I2P/COCO workflows.
- `scripts/style/` contains Van Gogh and Monet erase/retain workflows.
- `scripts/attack/` contains Ring-A-Bell evaluation.

## Installation

The reference environment is `environment.yaml`.

```bash
conda env create -f environment.yaml
conda activate SAGE
pip install -e .
cp scripts/config.env.example scripts/config.env
```

Use a CUDA Linux host for the complete training and evaluation workflow. The environment uses Python 3.8, PyTorch 2.1, CUDA 12.1, Diffusers 0.12.1, Transformers 4.25.1, NudeNet 3.4.2, and the OpenAI CLIP package used by Q16.

## Models and data

Weights and generated datasets are intentionally excluded. Place `sd-v1-4-full-ema.ckpt` under `models/`. Set `SD_MODEL_ID` to a Hub model ID or local Diffusers snapshot. Set `STYLE_CLASSIFIER_DIR`, `COCO_200_IMGS`, and `COCO_10K_IMGS` in `scripts/config.env`. Q16 prompt weights are included at `eval-scripts/finetuned_q16_prompts.pt`. The art style classifier is from UnlearnDiffAtk and you can download it [here](https://drive.google.com/file/d/1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po/view?usp=share_link).

Prompt CSV files must contain `case_number`, `prompt`, and `evaluation_seed`. Style retention CSV files also contain `artist`.

## Training

```bash
bash scripts/nudity/train.sh
STYLE=vangogh bash scripts/style/train.sh
```

Training writes checkpoints under `EXP_DIR/models/`. The default nudity run produces `TextEncoder-text_encoder_full-epoch_999.pt`.

## Generation

```bash
python generate-scripts/generate_images.py \
  --prompts_path data/prompts/eval/nudity_quick.csv \
  --folder_suffix nudity_quick \
  --save_path outputs/nudity_sage/generate \
  --target_ckpt outputs/nudity_sage/models/TextEncoder-text_encoder_full-epoch_999.pt \
  --checkpoint_component text_encoder --origin_or_target target --device cuda:0
```

Checkpoint component selection is inferred from state-dict keys and can be explicit with `--checkpoint_component`. Images are saved under `visualizations_<suffix>/SD-v1-4/<case_number>.png`. Additional samples use `<case_number>_1.png`, `<case_number>_2.png`, and so on. Cross-backbone generation is provided by `generate_morebaseline.py`.

## Evaluation

```bash
bash scripts/nudity/eval_quick.sh
RUN_Q16=1 bash scripts/nudity/eval_i2p.sh
STYLE=vangogh bash scripts/style/eval.sh
STYLE=monet bash scripts/style/eval.sh
```

The paper uses NudeNet threshold `0.6` and the five exposed-part classes in `eval-scripts/nudenet_classes.py`. The quick script defaults to threshold `0.5` for fast iteration. Use `--full_classes` to include the additional NudeNet labels. Q16 validates one image per `case_number` and writes aligned predictions and category rates to a sibling `<image-folder>_q16/` directory.

Style evaluation generates both prompt splits unless `ERASE_EDITED_PATH` and `RETAIN_EDITED_PATH` are supplied. Set `ERASE_ORIGINAL_PATH` and `RETAIN_ORIGINAL_PATH` to enable LPIPS.

## Model weights and checkpoints

The SAGE model weights of the relevant experiments are published in the [Hugging Face](https://huggingface.co/kevin1103/SAGE).

## Citation

If you use SAGE, please cite the corresponding SAGE paper and the upstream Stable Diffusion and evaluation projects.

```bibtex
@article{zhu2025sage,
  title={SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing},
  author={Zhu, Hongguang and Wei, Yunchao and Wang, Mengyu and Jiao, Siyu and Fang, Yan and Huang, Jiannan and Zhao, Yao},
  journal={arXiv preprint arXiv:2506.09363},
  year={2025}
}
```

## Acknowledgments

We thank the authors of Stable Diffusion, AdvUnlearn, UnlearnDiffAtk, NudeNet, Q16, RingABell and other evaluation projects for their contributions.