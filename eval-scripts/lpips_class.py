from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
import lpips
import cv2


# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


# def image_loader(image_name):
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     image = (image-0.5)*2
#     return image.to(torch.float)

def cal_lpips(df_prompts,original_path,edited_path):
    dist = []
    # dist_retain = []
    for index, row in df_prompts.iterrows(): 
        case_number = row.case_number
        file_name = f"{case_number}.png"
        img0 = lpips.load_image(os.path.join(original_path,file_name))
        img0 = cv2.resize(img0, (64, 64))
        img1 = lpips.load_image(os.path.join(edited_path,file_name))
        img1 = cv2.resize(img1, (64, 64))
        img0 = lpips.im2tensor(img0) # RGB image from [-1,1]
        img1 = lpips.im2tensor(img1)
        img0 = img0.cuda()
        img1 = img1.cuda()

        dist01 = loss_fn.forward(img0,img1)
        print(f'LPIPS score: {dist01.item()}')
        dist.append(dist01.item())

    return dist

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--erase_prompts_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--retain_prompts_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--erase_original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--retain_original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--erase_edited_path', help='path to edited image', type=str, required=True)
    parser.add_argument('--retain_edited_path', help='path to edited image', type=str, required=True)

    # parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    args = parser.parse_args()

    loss_fn = lpips.LPIPS(net='alex',version='0.1')
    loss_fn.cuda()
    # file_names = os.listdir(args.original_path) # read all the images in the original path
    # file_names = [name for name in file_names if '.png' in name]
    # df_prompts = pd.read_csv(args.prompts_path) # read the prompts csv to get correspoding case_number and prompts
    # df_prompts['lpips_loss'] = df_prompts['case_number'] *0 # initialise lpips column in df

    erase_original_path = args.erase_original_path
    erase_edited_path = args.erase_edited_path
    erase_prompts_path = args.erase_prompts_path

    retain_original_path = args.retain_original_path
    retain_edited_path = args.retain_edited_path
    retain_prompts_path = args.retain_prompts_path

    erase_df_prompts = pd.read_csv(erase_prompts_path) 
    retain_df_prompts = pd.read_csv(retain_prompts_path)

    dist_erase = cal_lpips(erase_df_prompts, erase_original_path, erase_edited_path)
    dist_retain = cal_lpips(retain_df_prompts, retain_original_path, retain_edited_path)

    avg_erase = np.mean(dist_erase)
    avg_retain = np.mean(dist_retain)

    print(f"Average LPIPS of erase: {sum(dist_erase)} / {len(dist_erase)} = {avg_erase}")
    print(f"Average LPIPS of retain: {sum(dist_retain)} / {len(dist_retain)} = {avg_retain}")
    print(f'lpips error = {avg_erase-avg_retain}')

          








    # name = folder.split('/')[-1]
    #     if name.strip() == '':
    #         name = folder.split('/')[-2]
    #     save_path = f'{folder[:-len(name)]}/{name}_NudeClasses_{int(threshold*100)}.csv'
    # out_file_path = os.path.join(opt.dir1, "lpips_against_sd14.txt")






    # loss_fn_alex = lpips.LPIPS(net='alex')
    
    # args = parser.parse_args()
    # file_names = os.listdir(args.original_path) # read all the images in the original path
    # file_names = [name for name in file_names if '.png' in name]
    # df_prompts = pd.read_csv(args.prompts_path) # read the prompts csv to get correspoding case_number and prompts
    
    # df_prompts['lpips_loss'] = df_prompts['case_number'] *0 # initialise lpips column in df
    # for index, row in df_prompts.iterrows(): 
    #     case_number = row.case_number
    #     files = [file for file in file_names if file.startswith(f'{case_number}_')]
    #     lpips_scores = []
    #     for file in files:
    #         print(file)
    #         # read both the files (original image to compare with and the edited image)
    #         original = image_loader(os.path.join(args.original_path,file))
    #         edited = image_loader(os.path.join(args.edited_path,file))
    #         # calculate lpips
    #         l = loss_fn_alex(original, edited)
    #         print(f'LPIPS score: {l.item()}')
    #         lpips_scores.append(l.item())
    #     df_prompts.loc[index,'lpips_loss'] = np.mean(lpips_scores)
    # if args.save_path is not None:
    #     if len(os.path.basename(args.edited_path).strip()) == 0:
    #         basename = args.edited_path.split('/')[-2]
    #     else:
    #         basename = args.edited_path.split('/')[-1]
    #     df_prompts.to_csv(os.path.join(args.save_path, f'{basename}_lpipsloss.csv'))

