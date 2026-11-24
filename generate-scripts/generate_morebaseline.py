"""Run cross-backbone inference using a separate Diffusers environment."""
import argparse
from pathlib import Path

MODELS = {
    'LCM_Dreamshaper': 'SimianLuo/LCM_Dreamshaper_v7',
    'openjourney': 'prompthero/openjourney-v4',
    'dreamlike': 'dreamlike-art/dreamlike-photoreal-2.0',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name', choices=MODELS, required=True)
    parser.add_argument('--model_id', help='Override with a local Diffusers snapshot or Hub model ID')
    parser.add_argument('--target_ckpt', help='Optional compatible edited text-encoder checkpoint')
    parser.add_argument('--prompts_path', required=True)
    parser.add_argument('--save_path', type=Path, required=True)
    parser.add_argument('--folder_suffix', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--from_case', type=int, default=0)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from diffusers import DiffusionPipeline
    from checkpoints import load_component_weights

    is_lcm = args.model_name == 'LCM_Dreamshaper'
    steps = args.ddim_steps if args.ddim_steps is not None else (4 if is_lcm else 50)
    pipeline = DiffusionPipeline.from_pretrained(args.model_id or MODELS[args.model_name])
    if args.target_ckpt:
        load_component_weights(args.target_ckpt, pipeline.text_encoder, component='text_encoder')
    pipeline.to(args.device)
    folder = args.save_path / args.folder_suffix / args.model_name
    folder.mkdir(parents=True, exist_ok=True)
    for row in pd.read_csv(args.prompts_path).itertuples(index=False):
        case = int(row.case_number)
        image_path = folder / f'{case}.png'
        if case < args.from_case or image_path.is_file():
            continue
        settings = {'lcm_origin_steps': 50} if is_lcm else {}
        result = pipeline(
            str(row.prompt),
            num_inference_steps=steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device='cpu').manual_seed(int(row.evaluation_seed)),
            **settings,
        )
        if getattr(result, 'nsfw_content_detected', None) and result.nsfw_content_detected[0]:
            raise RuntimeError(f'Pipeline flagged case {case}; use the method-specific evaluation outputs')
        result.images[0].save(image_path)


if __name__ == '__main__':
    main()
