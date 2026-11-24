from utils.util import *
from utils.attack_util import *
from utils.text_encoder import CustomTextEncoder
from utils.get_loss import *
from utils.prompt_dataset import *

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import random
import argparse
# import wandb
from pathlib import Path
import os
import copy

def SAGE(erase_w, retain_image_w, retain_text_w, prompt, dataset_retain, retain_batch, retain_batch_img, retain_train, retain_step , attack_method, train_method, norm_layer, component, start_guidance, negative_guidance, iterations, save_interval, lr, config_path, ckpt_path, diffusers_config_path, output_dir, devices, seperator=None, image_size=512, ddim_steps=50, adv_prompt_num=3, attack_embd_type='word_embd', attack_type='prefix_k', attack_init='latest', warmup_iter=200, attack_step=30, attack_lr=1e-2, adv_prompt_update_step=20):
    
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, n_samples=batch, till_T=t, verbose=False)
    
    # ====== Stage 0: PROMPT CLEANING ======
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(f'The Concept Prompt to be unlearned:{words}')
    
    retain_dataset = retain_prompt(dataset_retain)
    
    # ======= Stage 1: TRAINING SETUP =======
    ddim_eta = 0

    model_name_or_path = os.environ.get("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
    cache_path = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer", cache_dir=cache_path)
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_path).to(devices[0])
    CLIP_text_projection = clip.text_projection
    CLIP_text_projection.eval()
    CLIP_text_projection.requires_grad_(False)
    text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder", cache_dir=cache_path).to(devices[0])
    custom_text_encoder = CustomTextEncoder(text_encoder).to(devices[0])
    raw_text_encoder = CustomTextEncoder(copy.deepcopy(text_encoder)).to(devices[1])

    all_embeddings = custom_text_encoder.get_all_embedding().unsqueeze(0)

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)
    model_orig.eval()

    # Setup tainable model parameters
    if 'text_encoder' in train_method:
        parameters = param_choices(model=custom_text_encoder, train_method=train_method, component=component, final_layer_norm=norm_layer)
    else:
        parameters = param_choices(model=model, train_method=train_method, component=component, final_layer_norm=norm_layer)
    
    losses = []
    loss_retain_sims = []
    loss_erases = []
    opt = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.AdamW([{'params': model_params[key], 'lr':config['init_lr']} for key in model_params.keys()],weight_decay=config['weight_decay'])
    criteria = torch.nn.MSELoss()

    name = train_method

    # ========== Stage 2: Training ==========
    pbar = tqdm(range(iterations))
    global_step = 0
    attack_round = 0
    for i in pbar: 
        # Reset the dataset if all prompts are seen           
        if retain_dataset.check_unseen_prompt_count() < retain_batch:
            retain_dataset.reset()
        
        word = random.sample(words,1)[0]
        text_input = tokenizer(
            word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
        )
        text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
        
        # get conditional embedding for the prompt
        emb_0 = model_orig.get_learned_conditioning([''])
        emb_p = model_orig.get_learned_conditioning([word])

        raw_text_encoder.eval()
        raw_text_encoder.requires_grad_(False)

        if retain_train == 'reg':
            all_retain_words = retain_dataset.get_random_prompts(retain_batch)
            all_retain_text_input = tokenizer(
                all_retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
            )
            all_retain_input_ids = all_retain_text_input.input_ids.to(devices[0])
            all_retain_text_embeddings = id2embedding(tokenizer, all_embeddings, all_retain_input_ids, devices[0])
            all_retain_text_embeddings = all_retain_text_embeddings.reshape(retain_batch, -1, all_retain_text_embeddings.shape[-1])  # [batch, 77, 768]
            CLIP_text_projection.to(devices[1])
            raw_retain_text_embedding = raw_text_encoder(input_ids = all_retain_input_ids.to(devices[1]), inputs_embeds=all_retain_text_embeddings.to(devices[1]))[1].detach()
            raw_retain_text_feature = CLIP_text_projection(raw_retain_text_embedding).detach()


        # Change unlearned concept and obtain its corresponding adv embedding
        if i >= warmup_iter:
            # ===== ** Attack ** : get adversarial prompt
            custom_text_encoder.text_encoder.eval()
            custom_text_encoder.text_encoder.requires_grad_(False)
            model.eval()
            concept_type = 'style' if prompt=='vangogh' else 'object'
 
            attack_lr_n = attack_lr 
            print(attack_lr_n)

            attack_step_n = attack_step
            print(attack_step_n)

            adv_word_embd, adv_input_ids= prompt_self_augment(concept_type, CLIP_text_projection, word, tokenizer, raw_text_encoder, custom_text_encoder, devices, all_embeddings, attack_type, attack_step_n, attack_lr_n, attack_method)

            CLIP_text_projection.to(devices[0])
            global_step += attack_step
            attack_round += 1

        # Set model/TextEnocder to train or eval mode
        if 'text_encoder' in train_method:
            custom_text_encoder.text_encoder.train()
            custom_text_encoder.text_encoder.requires_grad_(True)
            model.eval()
        else:
            custom_text_encoder.text_encoder.eval()
            custom_text_encoder.text_encoder.requires_grad_(False)
            model.train()
        opt.zero_grad()
        
        # Retaining prompts for retaining regularized training
        if retain_train == 'reg':
            CLIP_text_projection.to(devices[0])
            now_retain_text_embedding = custom_text_encoder(input_ids = all_retain_input_ids.to(devices[0]), inputs_embeds=all_retain_text_embeddings.to(devices[0]))[1]
            now_retain_text_feature = CLIP_text_projection(now_retain_text_embedding)

            raw_score = sim_score(raw_retain_text_feature.to(devices[0]), raw_retain_text_feature.to(devices[0])) 
            now_score = sim_score(now_retain_text_feature.to(devices[0]), now_retain_text_feature.to(devices[0])) 
 

            loss_retain_sim_graph = criteria(raw_score,now_score)

            loss_retain_sim = loss_retain_sim_graph 

            if retain_batch_img>0:
                cross_score = sim_score(raw_retain_text_feature.to(devices[0]), now_retain_text_feature.to(devices[0]))
                diag_elements = torch.diag(cross_score)
                
                min_n_values, min_n_indices = torch.topk(diag_elements, retain_batch_img, largest=False)
                
                retain_words = [all_retain_words[idx] for idx in min_n_indices.tolist()]
                retain_input_ids = all_retain_input_ids[min_n_indices]
                retain_emb_p = model_orig.get_learned_conditioning(retain_words) #raw text encoder

                retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_input_ids, devices[0])
                retain_text_embeddings = retain_text_embeddings.reshape(retain_batch_img, -1, retain_text_embeddings.shape[-1])  
                retain_emb_n = custom_text_encoder(input_ids = retain_input_ids, inputs_embeds=retain_text_embeddings)[0] 
            else:
                retain_text_embeddings = None
                retain_emb_p = None
                retain_emb_n = None
                retain_train = 'esd'
        else:
            retain_text_input = None
            retain_text_embeddings = None
            retain_emb_p = None
            retain_emb_n = None
        
        if i < warmup_iter:
            # Warmup training
            input_ids = text_input.input_ids.to(devices[0])
            emb_n = custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
            loss_erase, loss_retain = get_train_loss_retain(retain_batch_img, retain_train, model, model_orig, custom_text_encoder, sampler, sampler_orig, emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, input_ids, attack_embd_type)
        else:
            # retain_train = 'esd'
            negative_guidance_now = negative_guidance #* H_score_2
            print(f'negative_guidance_now:{negative_guidance_now}')
            loss_erase, loss_retain = get_train_loss_retain(retain_batch_img, retain_train, model, model_orig, custom_text_encoder, sampler, sampler_orig, emb_0, emb_p, retain_emb_p, None, retain_emb_n, start_guidance, negative_guidance_now, devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_word_embd)
        
        loss = loss_erase * erase_w + loss_retain * retain_image_w + loss_retain_sim * retain_text_w
        # update weights to erase the concept
        loss.backward(retain_graph=True)
        losses.append(loss.item())
        loss_retain_sims.append(loss_retain_sim.item())
        loss_erases.append(loss_erase.item())
        pbar.set_postfix({"loss": loss.item(), 
                          "loss_erase":loss_erase.item(),
                          "loss_retain":loss_retain.item(),
                          "loss_retain_sim": loss_retain_sim.item(),
                          "loss_retain_sim_graph": loss_retain_sim_graph.item(),
                          })

        # wandb.log({'Train_Loss':loss.item()}, step=global_step)
        # wandb.log({'Attack_Loss': 0.0}, step=global_step)
        global_step += 1
        
        opt.step()
        
        # ====== Stage 3: save final model and loss curve ======
        # save checkpoint and loss curve
        # if (i+1) % save_interval == 0 and i+1 != iterations and i+1>= save_interval:
        #     if 'text_encoder' in train_method:
        #         save_text_encoder(output_dir, custom_text_encoder, name, i)
        #     else:
        #         save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

        if i % 100 == 0:
            save_history(output_dir, losses, word_print, loss_name='all')
            save_history(output_dir, loss_erases, word_print, loss_name='erase')
            save_history(output_dir, loss_retain_sims, word_print, loss_name='retain_sim')

    # Save final model and loss curve
    model.eval()
    custom_text_encoder.text_encoder.eval()
    custom_text_encoder.text_encoder.requires_grad_(False)
    if 'text_encoder' in train_method:
        save_text_encoder(output_dir, custom_text_encoder, name, i)
    else:
        save_model(output_dir, model, name, i, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(output_dir, losses, word_print, loss_name='all')
    save_history(output_dir, loss_erases, word_print, loss_name='erase')
    save_history(output_dir, loss_retain_sims, word_print, loss_name='retain_sim')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'SAGE',
                    description = 'SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing')

    # Diffusion setup
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='models/diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    
    # Training setup
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=False, default='nudity')
    parser.add_argument('--dataset_retain', help='prompts corresponding to non-target concept to retain', type=str, required=False, default='coco_object', choices=['coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'])
    parser.add_argument('--retain_batch_img', help='batch size of unlearning prompt during training', type=int, required=False, default=1)
    parser.add_argument('--retain_batch', help='batch size of retaining prompt during training', type=int, required=False, default=2)
    parser.add_argument('--retain_train', help='different retaining version: reg (regularization) or iter (iterative)', type=str, required=False, default='iter', choices=['iter', 'reg'])
    parser.add_argument('--retain_step', help='number of steps for retaining prompts', type=int, required=False, default=1)
    parser.add_argument('--retain_image_w', help='retaining loss weight', type=float, required=False, default=1.0)
    parser.add_argument('--retain_text_w', help='retaining loss weight', type=float, required=False, default=1.0)
    parser.add_argument('--erase_w', help='retaining loss weight', type=float, required=False, default=1.0)
    
    parser.add_argument('--train_method', help='method of training', type=str, choices=['text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01', 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234', 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567', 'text_encoder_layer012345678', 'text_encoder_layer0123456789', 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011', 'text_encoder_layer0_11','text_encoder_layer01_1011', 'text_encoder_layer012_91011', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer'], default='text_encoder_full', required=False)
    parser.add_argument('--norm_layer', help='During training, norm layer to be updated or not', action='store_true', default=False, required=False)
    parser.add_argument('--attack_method', help='method of training', type=str, choices=['pgd', 'multi_pgd', 'fast_at', 'free_at'], default='pgd', required=False)
    parser.add_argument('--component', help='component', type=str, choices=['all', 'ffn', 'attn'], default='all', required=False)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--save_interval', help='iterations used to train', type=int, required=False, default=200)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    
    # Attack hyperparameters
    parser.add_argument('--adv_prompt_num', help='number of prompt token for adversarial soft prompt learning', type=int, required=False, default=1)
    parser.add_argument('--attack_embd_type', help='the adversarial embd type: word embedding, condition embedding', type=str, required=False, default='word_embd', choices=['word_embd', 'condition_embd'] )
    parser.add_argument('--attack_type', help='the attack type: append or add', type=str, required=False, default='prefix_k', choices=['replace_k' ,'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'])
    parser.add_argument('--attack_init', help='the attack init: random or latest', type=str, required=False, default='latest', choices=['random', 'latest'])
    parser.add_argument('--attack_step', help='adversarial attack steps', type=int, required=False, default=30)
    parser.add_argument('--adv_prompt_update_step', help='after every n step, adv prompt would be updated', type=int, required=False, default=1)
    parser.add_argument('--attack_lr', help='learning rate used to train', type=float, required=False, default=1e-3)
    parser.add_argument('--warmup_iter', help='the number of warmup interations before attack', type=int, required=False, default=200)
    
    # Log details
    parser.add_argument('--result_path', help='wandb project name', type=str, required=True, default='SAGE')

    args = parser.parse_args()
    
    prompt = args.prompt  
    dataset_retain = args.dataset_retain
    retain_batch = args.retain_batch
    retain_train = args.retain_train
    retain_step = args.retain_step
    retain_batch_img = args.retain_batch_img
    retain_image_w = args.retain_image_w
    retain_text_w = args.retain_text_w
    erase_w = args.erase_w
    
    train_method = args.train_method
    norm_layer = args.norm_layer
    attack_method = args.attack_method
    component = args.component
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    save_interval = args.save_interval
    lr = args.lr
    
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    
    adv_prompt_num = args.adv_prompt_num
    attack_embd_type = args.attack_embd_type
    attack_type = args.attack_type
    attack_init = args.attack_init
    attack_step = args.attack_step
    attack_lr = args.attack_lr
    adv_prompt_update_step = args.adv_prompt_update_step
    warmup_iter = args.warmup_iter
    
    # Directory setup
    # experiment_name = f'AdvUnlearn-{prompt}-method_{train_method}_{component}-Attack_{args.attack_method}-Retain_{dataset_retain}_{retain_train}_{retain_loss_w}-lr_{lr}-AttackLr_{attack_lr}-{attack_type}_adv_num_{adv_prompt_num}-{attack_embd_type}-attack_init_{attack_init}-attack_step_{attack_step}-adv_update_{adv_prompt_update_step}-warmup_iter_{warmup_iter}'
    
    # run_dir = Path("./results/results_with_retaining") / args.prompt / args.dataset_retain / args.attack_method / f'AttackLr_{args.attack_lr}' / args.train_method / f'{args.component}' / args.attack_type / experiment_name / "wandb_logs"
    # output_dir = Path("./results/results_with_retaining") / args.prompt / args.dataset_retain / args.attack_method /  f'AttackLr_{args.attack_lr}' / args.train_method / f'{args.component}' / args.attack_type / experiment_name

    
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    output_dir = Path(args.result_path)
    if not output_dir.exists():
        os.makedirs(str(output_dir))
        
    # wandb.init(config=args,
    #            project=args.project_name,
    #            name=experiment_name,
    #            dir=str(run_dir),
    #            reinit=True)

    print_args_table(parser, args)
    
    SAGE(erase_w=erase_w, retain_image_w=retain_image_w, retain_text_w=retain_text_w, prompt=prompt, dataset_retain=dataset_retain, retain_batch=retain_batch, retain_batch_img=retain_batch_img,retain_train=retain_train, retain_step=retain_step,  attack_method = attack_method, train_method=train_method, norm_layer=norm_layer, component=component, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, save_interval=save_interval, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, output_dir= output_dir, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, adv_prompt_num=adv_prompt_num, attack_embd_type=attack_embd_type, attack_type=attack_type, attack_init=attack_init, warmup_iter=warmup_iter, attack_step=attack_step, attack_lr=attack_lr, adv_prompt_update_step=adv_prompt_update_step)
