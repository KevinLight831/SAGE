from transformers import pipeline
import argparse
import glob
from PIL import Image
import pandas as pd
import os
import torch
import torchvision.transforms as transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # desired size of the output image
# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# loader = transforms.Compose([
#     transforms.Resize(imsize),  # scale imported image
#     transforms.ToTensor()])  # transform it into a torch tensor


# def image_loader(image_name):
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)


def init_classifier(device,path):
    return pipeline('image-classification',model=path,device=device)

def style_eval(classifier,img):
    return classifier(img,top_k=129)

def style_rank(data, artist):
    df = pd.DataFrame(data)

    # 按照score列进行降序排序
    df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)

    # 找到label为vincent-van-gogh的排名
    rank = df_sorted[df_sorted['label'] == artist].index[0] + 1

    return rank




if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Style Classes',
                    description = 'Get Nudenet classes for images')
    parser.add_argument('--erase_folder', help='folder path to erased-style images', type=str, required=True)
    parser.add_argument('--retain_folder', help='folder path to retained-style images', type=str, required=True)
    parser.add_argument('--prompts_path', help='erase prompts csv', type=str, required=False, default='data/prompts/eval/vangogh.csv')
    parser.add_argument('--retain_prompts_path', help='retain prompts csv', type=str, required=False, default='data/prompts/eval/other_artists.csv')
    parser.add_argument('--classifier_dir', help='path to style classifier checkpoint', type=str, required=True)
    parser.add_argument('--erase_artist', help='target artist label in classifier', type=str, required=False, default='vincent-van-gogh')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    args = parser.parse_args()
    
    erase_folder = args.erase_folder
    retain_folder = args.retain_folder
    classifier_dir = args.classifier_dir
    device = args.device
    prompts_path = args.prompts_path
    retain_prompts_path = args.retain_prompts_path
    erase_artist = args.erase_artist

    classifier = init_classifier(device,classifier_dir)

    print(erase_folder)

    results = {}
    erase_df = pd.read_csv(prompts_path)
    total_len = len(erase_df)
    score = 0
    number = 0
    rank_erase = 0
    for index, row in erase_df.iterrows():
        case_number = row.case_number
        image_name = f"{erase_folder}/{case_number}.png"
        results['image'] = Image.open(image_name)
        results['style'] = style_eval(classifier,results['image'])

        rank = style_rank(results['style'], erase_artist)

        n = len(results['style'])
        rank_p = (n - rank) / (n - 1)
        rank_erase += rank_p

        success = erase_artist in list(map(lambda x: x['label'], results['style'][:3]))
        # print(results['style'])
        if success:
            number+=1

    avg_number_erase = number/total_len#越小越好
    avg_rank_erase = rank_erase/total_len #越小越好

    retain_df = pd.read_csv(retain_prompts_path)
    total_len = len(retain_df)
    score = 0
    number = 0
    rank_retain = 0
    for index, row in retain_df.iterrows():
        case_number = row.case_number
        image_name = f"{retain_folder}/{case_number}.png"
        results['image'] = Image.open(image_name)
        results['style'] = style_eval(classifier,results['image'])
        
        rank = style_rank(results['style'], row.artist) #rank没再使用了

        n = len(results['style'])
        rank_p = (n - rank) / (n - 1)
        rank_retain += rank_p

        success = row.artist in list(map(lambda x: x['label'], results['style'][:3]))
        if success:
            number+=1
        else:
            print(case_number)

    avg_number_retain = number/total_len #越大越好
    avg_rank_retain = rank_retain/total_len #越大越好

    print(f'avg_number_erase = {avg_number_erase}')
    # print(f'avg_rank_erase = {avg_rank_erase}')

    print(f'avg_number_retain = {avg_number_retain}')
    # print(f'avg_rank_retain = {avg_rank_retain}')

    # F_number= 2.0/(1.0/(1-avg_number_erase) + 1.0avg_number_erase/(avg_number_retain))
    F_number= avg_number_retain - avg_number_erase
    print(f'F_number = {F_number}')
    # F_rank= 2.0/(1.0/(1-avg_rank_erase) + 1.0/(avg_rank_retain))
    # print(f'F_rank = {F_rank}')


