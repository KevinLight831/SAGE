from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
from sld import SLDPipeline
import os
from accelerate import PartialState, Accelerator

def generate_SLD(sld_concept, sld_type,  prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=5, from_case=0):
    '''
    Generates Images with SLD pipeline

    Parameters
    ----------
    sld_concept : str
        The concept to be considered safe.
    sld_type : str
        The settings for SLD to use (Medium, Max, Weak).
    prompts_path : str
        Path to the csv with prompts.
    save_path : str
        Path to the folder to store the images.
    device : str, optional
        Device to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        Guidance value to run classifier free guidance. The default is 7.5.
    image_size : int, optional
        Size of the image to generate. The default is 512.
    ddim_steps : int, optional
        Number of diffusion steps. The default is 100.
    num_samples : int, optional
        Number of images to be generated per prompt. The default is 5.
    from_case : int, optional
        offset for the images to be generated from csv. The default is 0.

    Returns
    -------
    None.

    '''
    # accelerator = Accelerator()
    # state = PartialState()
    # pipe = SLDPipeline.from_pretrained(os.environ.get("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")).to(state.device)
    # pipe.safety_checker = None
    # pipe.requires_safety_checker = False
    # gen = torch.Generator(device=state.device)

    pipe = SLDPipeline.from_pretrained(os.environ.get("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    gen = torch.Generator(device=device)
    # if sld concept is different from default, replace the concept in the pipe
    if sld_concept is not None:
        pipe.safety_concept = sld_concept
    
    print(pipe.safety_concept)
    folder_path = f"{save_path}/SLD_{sld_type}"
    os.makedirs(folder_path, exist_ok=True)
    
    df = pd.read_csv(prompts_path)

    import time
    a = time.time()

    # step = accelerator.num_processes
    # for i in range(0, len(df), step): 
    #     idx = i + state.process_index
    #     if idx < len(df):
    #         row = df.iloc[idx]

    for _, row in df.iterrows():
        prompt = str(row.prompt)
        seed = int(row.evaluation_seed)
        case_number = row.case_number
        if case_number<from_case:
            continue

        gen.manual_seed(seed)
        if sld_type is not None:
            if sld_type == 'Medium':
                sld_guidance_scale = 1000
                sld_warmup_steps = 10
                sld_threshold = 0.01
                sld_momentum_scale = 0.3
                sld_mom_beta = 0.4
            if sld_type == 'Max':
                sld_guidance_scale = 5000
                sld_warmup_steps = 0
                sld_threshold = 1.0
                sld_momentum_scale = 0.5
                sld_mom_beta = 0.7
            if sld_type == 'Weak':
                sld_guidance_scale = 200
                sld_warmup_steps = 15
                sld_threshold = 0.0
                sld_momentum_scale = 0.0
                sld_mom_beta = 0.0
        if os.path.isfile(os.path.join(folder_path, f"{case_number}.png")):
            print(f"File {case_number}.png exists.")
            continue

        out = pipe(prompt=prompt, generator=gen, seed=seed, guidance_scale=guidance_scale, height=image_size, width=image_size, num_inference_steps=ddim_steps, num_images_per_prompt=num_samples, sld_guidance_scale = sld_guidance_scale, sld_warmup_steps = sld_warmup_steps, sld_threshold = sld_threshold, sld_momentum_scale = sld_momentum_scale, sld_mom_beta = sld_mom_beta).images

        for num, im in enumerate(out):
            im.save(f"{folder_path}/{case_number}{f'_{num}' if num else ''}.png")
        # accelerator.wait_for_everyone()
    print(time.time()-a)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--sld_concept', help='concept to remove from SLD', type=str, required=False, default =None)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--sld_type', help='Type of SLD', choices=['Medium', 'Max', 'Weak'], default='Medium')
    args = parser.parse_args()
    
    sld_concept = args.sld_concept
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    sld_type = args.sld_type
    generate_SLD(sld_concept, sld_type, prompts_path, save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case)
