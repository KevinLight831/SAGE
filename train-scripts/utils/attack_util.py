import torch
from .util import *
import wandb

style_prompts = [
    "An artwork by {}",
    "Art piece by {}",
    "A recent creation by {}",
    "{}'s renowned art",
    "Latest masterpiece by {}",
    "A stunning image by {}",
    "An art in {}'s style",
    "Exhibition artwork of {}",
    "Art display by {}",
    "A beautiful painting by {}",
    "An image inspired by {}'s style",
    "A sketch by {}",
    "Art piece representing {}",
    "A drawing by {}",
    "Artistry showcasing {}",
    "An illustration by {}",
    "A digital art by {}",
    "A visual art by {}",
    "A reproduction inspired by {}'s colorful, expressive style",
    "Famous painting of {}",
    "A famous art by {}",
    "Artistic style of {}",
    "{}'s famous piece",
    "Abstract work of {}",
    "{}'s famous drawing",
    "Art from {}'s early period",
    "A portrait by {}",
    "An imitation reflecting the style of {}",
    "An painting from {}'s collection",
    "Vibrant reproduction of artwork by {}",
    "Artistic image influenced by {}",
]

object_prompts =  [
    "{} in a photo",
    "{} in a snapshot",
    "A snapshot of {}",
    "A photograph showcasing {}",
    "An illustration of {}",
    "A digital rendering of {}",
    "A visual representation of {}",
    "A graphic of {}",
    "A shot of {}",
    "A photo of {}",
    "A black and white image of {}",
    "A depiction in portrait form of {}",
    "A scene depicting {} during a public gathering",
    "{} captured in an image",
    "A depiction created with oil paints capturing {}",
    "An image of {}",
    "A drawing capturing the essence of {}",
    "An official photograph featuring {}",
    "A detailed sketch of {}",
    "{} during sunset/sunrise",
    "{} in a detailed portrait",
    "An official photo of {}",
    "Historic photo of {}",
    "Detailed portrait of {}",
    "A painting of {}",
    "HD picture of {}",
    "Magazine cover capturing {}",
    "Painting-like image of {}",
    "Hand-drawn art of {}",
    "An oil portrait of {}",
    "{} in a sketch painting",
]


def sim_score(image_embeds, text_embeds):
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    score = torch.matmul(text_embeds, image_embeds.t())
    return score

def split_id_embed(input_ids, input_embed, cot_start, cot_end, eot_index):
    id_embd_dict = {}
    id_embd_dict['sot_id'] = input_ids[:, :1] 
    id_embd_dict['pot_id'] = torch.cat([input_ids[:, 1:cot_start], input_ids[:, cot_end:eot_index]], dim=1) 
    id_embd_dict['cot_id'] = input_ids[:, cot_start:cot_end] #nudity word
    id_embd_dict['eot_id'] = input_ids[:, eot_index:] #pad tokens
    id_embd_dict['sot_embd']  = input_embed[:, :1]
    id_embd_dict['pot_embd'] = torch.cat([input_embed[:, 1:cot_start], input_embed[:, cot_end:eot_index]], dim=1)
    id_embd_dict['cot_embd'] = input_embed[:, cot_start:cot_end]
    id_embd_dict['eot_embd']  = input_embed[:, eot_index:]
    return id_embd_dict

def construct_id_embed(id_embd_dict, cot_start, adv_embedding):
    adv_start = cot_start-1
    input_ids = torch.cat([id_embd_dict['sot_id'], 
                            id_embd_dict['pot_id'][:, :adv_start], 
                            id_embd_dict['cot_id'], 
                            id_embd_dict['pot_id'][:, adv_start:], 
                            id_embd_dict['eot_id']], 
                            dim=1)

    embed = torch.cat([id_embd_dict['sot_embd'],
                        adv_embedding[:, :adv_start], 
                        id_embd_dict['cot_embd'],
                        adv_embedding[:, adv_start:], 
                        id_embd_dict['eot_embd']], 
                        dim=1)
    return input_ids, embed

def split_id_embed_all(input_ids, input_embed, cot_start, cot_end, eot_index):
    id_embd_dict = {}
    id_embd_dict['sot_id'] = input_ids[:, :1] 
    id_embd_dict['pot_id'] = input_ids[:, 1:eot_index]
    id_embd_dict['eot_id'] = input_ids[:, eot_index:] 
    id_embd_dict['sot_embd']  = input_embed[:, :1]
    id_embd_dict['pot_embd'] = input_embed[:, 1:eot_index]
    id_embd_dict['eot_embd']  = input_embed[:, eot_index:]
    return id_embd_dict

def construct_id_embed_all(id_embd_dict, cot_start, adv_embedding):
    adv_start = cot_start-1
    input_ids = torch.cat([id_embd_dict['sot_id'], 
                            id_embd_dict['pot_id'], 
                            id_embd_dict['eot_id']], 
                            dim=1)

    embed = torch.cat([id_embd_dict['sot_embd'],
                        adv_embedding,
                        id_embd_dict['eot_embd']], 
                        dim=1)
    return input_ids, embed



def init_adv(k, tokenizer, all_embeddings, attack_type, device, batch = 1, attack_init_embd = None):
    # Different attack types have different initializations (Attack types: add, insert)
    adv_embedding = torch.nn.Parameter(torch.randn([batch, k, 768])).to(device)
    
    if attack_init_embd is not None:
        # Use the provided initial adversarial embedding
        adv_embedding.data = attack_init_embd[:,1:1+k].data
    else:
        # Random sample k words from the vocabulary as the initial adversarial words
        tmp_ids = torch.randint(0,len(tokenizer),(batch, k)).to(device) 
        tmp_embeddings = id2embedding(tokenizer, all_embeddings, tmp_ids, device)
        tmp_embeddings = tmp_embeddings.reshape(batch, k, 768)
        adv_embedding.data = tmp_embeddings.data 
    adv_embedding = adv_embedding.detach().requires_grad_(True)
    
    return adv_embedding


def prompt_self_augment(concept_type, CLIP_text_projection, word, tokenizer, raw_text_encoder, text_encoder, devices, all_embeddings, attack_type,  attack_step, attack_lr,  attack_method='pgd'):
    
    '''
    Perform soft prompt attack on the ESD model
    Args:
        attack_type: str
            The type of attack (add or insert)
        attack_embd_type: str
            The type of adversarial embedding (condition_embd or word_embd)
        attack_step: int
            The number of steps for the attack
        attack_lr: float
            The learning rate for the attack
        attack_init: str
            The initialization method for the attack (latest or random)
        attack_init_embd: torch.Tensor
            The initial adversarial embedding
    '''
    orig_prompt_len = len(word.split())
    if attack_type == 'add':
        k = orig_prompt_len
      
    if concept_type == 'style':
        prompt_template = random.choice(style_prompts)
    else:
        prompt_template = random.choice(object_prompts)
    sentence = prompt_template.format(word)


    text_input = tokenizer(
        sentence, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True
    )
    input_ids = text_input.input_ids.to(devices[1])
    word_embedding = id2embedding(tokenizer, all_embeddings.to(devices[1]), input_ids, devices[1])

    CLIP_text_projection.to(devices[1])
    raw_text_feature = CLIP_text_projection(raw_text_encoder(input_ids = input_ids, inputs_embeds=word_embedding.to(devices[1]))[1]).detach() 
    CLIP_text_projection.to(devices[0])
    now_raw_text_feature = CLIP_text_projection(text_encoder(input_ids = input_ids.to(devices[0]), inputs_embeds=word_embedding.to(devices[0]))[1]).detach() 

    word_tokens = tokenizer(word, add_special_tokens=False)["input_ids"]

    word_start_index = -1
    for i in range(len(input_ids[0]) - len(word_tokens) + 1):
        if (input_ids[0][i:i + len(word_tokens)] == torch.tensor(word_tokens).to(devices[1])).all():
            word_start_index = i
            break
    word_end_index = word_start_index + len(word_tokens)

    template_length = (input_ids[0] != tokenizer.pad_token_id).sum().item()


    id_embd_dict = split_id_embed(input_ids, word_embedding, word_start_index, word_end_index, template_length)
    adv_embedding = torch.nn.Parameter(id_embd_dict['pot_embd'].clone().detach(), requires_grad=True).to(devices[1])
    attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr) 

    for i in range(attack_step):
        adv_input_ids, input_adv_word_embedding = construct_id_embed(id_embd_dict, word_start_index, adv_embedding)
        now_adv_text_output = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding.to(devices[0]))
        raw_adv_text_output = raw_text_encoder(input_ids = adv_input_ids.to(devices[1]), inputs_embeds=input_adv_word_embedding.to(devices[1]))
        now_adv_condition_embedding, now_adv_pooled_output = now_adv_text_output[0], now_adv_text_output[1]
        raw_adv_condition_embedding, raw_adv_pooled_output = raw_adv_text_output[0], raw_adv_text_output[1]
        CLIP_text_projection.to(devices[1])
        raw_adv_text_feature = CLIP_text_projection(raw_adv_pooled_output) #原始的textencoder(adv)
        CLIP_text_projection.to(devices[0])
        now_adv_text_feature = CLIP_text_projection(now_adv_pooled_output).to(devices[1]) #当前的textencoder(adv) 


        H_test = sim_score(raw_text_feature, now_raw_text_feature)

        # Calculate H1 = the original textencoder(text) and the original textencoder(adv). the similarity should be reduced
        H_score_1 = sim_score(raw_text_feature, raw_adv_text_feature)

        # Calculate H2 = the original textencoder(text) and the current textencoder(adv). the similarity should be increased
        H_score_2 = sim_score(raw_text_feature, now_adv_text_feature)
        if i==0:
            H_score_2_raw = H_score_2
        
        print(f"{i}step: H_test:{H_test.item()}, H_score_1:{H_score_1.item()}, H_score_2:{H_score_2.item()}, H_score_2_raw:{H_score_2_raw.item()}")
        loss = -H_score_2 + H_score_1/H_score_2 #In the paper, H1 is H2 here, and H2 is H1 here
        
        print(loss.item())
        loss.backward()
        
        if attack_method == 'pgd':
            attack_opt.step()
        elif attack_method == 'fast_at':
            adv_embedding.grad.sign_()
            attack_opt.step()
        else:
            raise ValueError('attack_method must be either pgd or fast_at')
        
    return input_adv_word_embedding.to(devices[0]), adv_input_ids.to(devices[0])


        
def split_embd(input_embed, k, orig_prompt_len):
    sot_embd, mid_embd, replace_embd, eot_embd = torch.split(input_embed, [1, orig_prompt_len, k, 76-orig_prompt_len-k ], dim=1)
    return sot_embd, mid_embd, replace_embd, eot_embd
    
def split_id(input_ids, k, orig_prompt_len):
    sot_id, mid_id, replace_id, eot_id = torch.split(input_ids, [1, orig_prompt_len, k, 76-orig_prompt_len-k], dim=1)
    return sot_id, mid_id, replace_id, eot_id

def construct_embd(k, adv_embedding, insertion_location, sot_embd, mid_embd, eot_embd):
    if insertion_location == 'prefix_k':     # Prepend k words before the original prompt
        embedding = torch.cat([sot_embd,adv_embedding,mid_embd,eot_embd],dim=1)
    elif insertion_location == 'replace_k':  # Replace k words in the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,mid_embd.shape[1],1)
        embedding = torch.cat([sot_embd,adv_embedding,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'add':      # Add perturbation to the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,k,1)
        embedding = torch.cat([sot_embd,adv_embedding+mid_embd,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'suffix_k':   # Append k words after the original prompt
        embedding = torch.cat([sot_embd,mid_embd,adv_embedding,eot_embd],dim=1)
    elif insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        embedding.append(mid_embd[:,:total_num//2,:])
        embedding.append(adv_embedding)
        embedding.append(mid_embd[:,total_num//2:,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    elif insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            embedding.append(mid_embd[:,internals*i:internals*(i+1),:])
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
        embedding.append(mid_embd[:,internals*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
        
    elif insertion_location == 'per_k_words':
        embedding = [sot_embd,]
        for i in range(adv_embedding.size(1) - 1):
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(mid_embd[:,3*i:3*(i+1),:])
        embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
        embedding.append(mid_embd[:,3*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    return embedding

def construct_id(k, adv_id, insertion_location,sot_id,eot_id,mid_id):
    if insertion_location == 'prefix_k':
        input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        
    elif insertion_location == 'replace_k':
        replace_id = eot_id[:,0].repeat(1,mid_id.shape[1])
        input_ids = torch.cat([sot_id,adv_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'add':
        replace_id = eot_id[:,0].repeat(1,k)
        input_ids = torch.cat([sot_id,mid_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'suffix_k':
        input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        
    elif insertion_location == 'mid_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        input_ids.append(mid_id[:,:total_num//2])
        input_ids.append(adv_id)
        input_ids.append(mid_id[:,total_num//2:])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'insert_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            input_ids.append(mid_id[:,internals*i:internals*(i+1)])
            input_ids.append(adv_id[:,i].unsqueeze(1))
        input_ids.append(mid_id[:,internals*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'per_k_words':
        input_ids = [sot_id,]
        for i in range(adv_id.size(1) - 1):
            input_ids.append(adv_id[:,i].unsqueeze(1))
            input_ids.append(mid_id[:,3*i:3*(i+1)])
        input_ids.append(adv_id[:,-1].unsqueeze(1))
        input_ids.append(mid_id[:,3*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
    return input_ids