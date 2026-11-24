from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os

from checkpoints import load_component_weights


@torch.inference_mode()
def generate_images(model_name, prompts_path, target_ckpt, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, folder_suffix='imagenette', origin_or_target='target', checkpoint_component='auto'):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'evaluation_seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    model_map = {
        'SD-v1-4': os.environ.get('SD_MODEL_ID', 'CompVis/stable-diffusion-v1-4'),
        'SD-V2': 'stabilityai/stable-diffusion-2-base',
        'SD-V2-1': 'stabilityai/stable-diffusion-2-1-base',
    }
    dir_ = model_map.get(model_name, model_map['SD-v1-4'])
    cache_path = os.environ.get('HF_HOME', None)

    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae", cache_dir=cache_path)
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer", cache_dir=cache_path)
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder", cache_dir=cache_path)
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet", cache_dir=cache_path)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)


    if origin_or_target == 'target':
        load_component_weights(target_ckpt, text_encoder, unet, checkpoint_component)
    save_path = f'{save_path}/visualizations_{folder_suffix}'

    vae.to(device).eval()
    text_encoder.to(device).eval()
    unet.to(device).eval()
    torch_device = device
    df = pd.read_csv(prompts_path)

    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    import time
    a = time.time()
    for _, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number<from_case:
            continue
        filenames = [f"{case_number}.png"] + [
            f"{case_number}_{index}.png" for index in range(1, num_samples)
        ]
        if all(os.path.isfile(os.path.join(folder_path, name)) for name in filenames):
            print(f"File {case_number}.png exists.")
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.Generator(device="cpu").manual_seed(int(seed))        # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(os.path.join(folder_path, filenames[num]))
    print(time.time()-a)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=False, default='SD-v1-4')
    parser.add_argument('--target_ckpt', help='path to unlearned checkpoint')
    parser.add_argument('--checkpoint_component', choices=['auto', 'text_encoder', 'unet'], default='auto')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--folder_suffix', help='folder dir suff ix', type=str, required=True)
    parser.add_argument('--origin_or_target', help='origin or target', choices=['origin', 'target'], default='target')
    args = parser.parse_args()
    if args.origin_or_target == 'target' and not args.target_ckpt:
        parser.error('--target_ckpt is required in target mode')
    if args.num_samples < 1:
        parser.error('--num_samples must be positive')
    
    model_name = args.model_name
    target_ckpt = args.target_ckpt
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    folder_suffix = args.folder_suffix
    origin_or_target = args.origin_or_target
    
    generate_images(model_name, prompts_path, target_ckpt, save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, folder_suffix = folder_suffix, origin_or_target=origin_or_target, checkpoint_component=args.checkpoint_component)
