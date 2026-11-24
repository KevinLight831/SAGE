#maces生成模型所用版本不同，必须新开一个py和对应环境mace。
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler

def extract_text_encoder_ckpt(ckpt_path):
    full_ckpt = torch.load(ckpt_path)
    new_ckpt = {}
    for key in full_ckpt.keys():
        if 'text_encoder.text_model' in key:
            new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
    return new_ckpt

def generate_images(model_name, prompts_path, target_ckpt, save_path,  device='cuda:0', guidance_scale = 7.5,  ddim_steps=100, num_samples=1, from_case=0, folder_suffix='imagenette', origin_or_target='target'):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
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

    save_path = f'{save_path}/visualizations_{folder_suffix}'
    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(target_ckpt)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.to(device)

    df = pd.read_csv(prompts_path)

    import time
    a = time.time()    
    for _, row in df.iterrows():    
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number<from_case:
            continue

        images = pipe(prompt, 
                      num_inference_steps=ddim_steps, 
                      guidance_scale=guidance_scale, 
                      num_images_per_prompt=1, 
                      generator=torch.manual_seed(seed)).images
        for num, im in enumerate(images):
            im.save(f"{folder_path}/{case_number}{f'_{num}' if num else ''}.png")   
    print(time.time()-a)



if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=False, default='SD-v1-4')
    parser.add_argument('--target_ckpt', help='path to unlearned CKPT', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--folder_suffix', help='folder dir suff ix', type=str, required=True)
    parser.add_argument('--origin_or_target', help='origin or target', type=str, required=False, default='target')
    args = parser.parse_args()
    
    model_name = args.model_name
    target_ckpt = args.target_ckpt
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    folder_suffix = args.folder_suffix
    origin_or_target = args.origin_or_target
    
    generate_images(model_name, prompts_path, target_ckpt, save_path,  device=device,
                    guidance_scale = guidance_scale, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, folder_suffix = folder_suffix, origin_or_target=origin_or_target)
