import post_process as pp
import pandas as pd
import csv
from pathlib import Path
import torch
import logging
from torchvision.utils import save_image
from datasets_resnet import *
import sys
import os
from torch.autograd import Variable
import cv2 as cv
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import sys
import logging
import shutil

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

def eval_data(args, generator_unet, data_loader, Tensor):

    generator_unet.eval()
    mse_loss = torch.nn.MSELoss()
    input_shape = (args.n_channel, args.img_height, args.img_width)
    sum_test = 0
    tmp_path = "./tmp/user_cv.jpg"
    tmp_csv = "./tmp/user_result.csv"
    json_dir = "./tmp/user_json"
    jpg_dir = "./tmp/user_jpg"
    png_dir = "./tmp/user_png"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    clean_dir(json_dir)
    clean_dir(jpg_dir)
    result = []

    # validation_transform = transforms.Compose(
    #         [
    #             transforms.Resize(input_shape[-2:], Image.BICUBIC),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         ]
    #     )

    for i, batch in enumerate(data_loader):
        
        real_A = Variable(batch["A"].type(Tensor))
        noisy = Variable(batch["aug_A"].type(Tensor))
        generated, e, e1, e2 = generator_unet(real_A)
        _, z, z1, z2 = generator_unet(noisy)
        error = mse_loss(e,z) + mse_loss(e1,z1) + mse_loss(e2,z2)
        save_image(generated.detach().cpu()[0],tmp_path,normalize=True)
        img_cv = Image.open(tmp_path)
        img_cv.load()
        img_cv = img_cat = np.asarray(img_cv, dtype='uint8')
        tp, unc, area_p, cnts =pp.critic_segmentation(img_cv)
        #tp = 2 if unc > args.uncertainty else tp

        #img_cv = save_image(real_A.detach().cpu()[0],tmp_path)
        img_cv = Image.open(os.path.relpath("".join(batch["pathA"])))
        img_cv.load()
        img_cv = np.asarray(img_cv, dtype='uint8')
        img_cat = cv.vconcat([img_cv, img_cat])
        
        pathA = batch["pathA"]
        # pathB = batch["pathB"]
        base = os.path.splitext(pathA[0])[0]
        _, basename = os.path.split(base)
        jsonpath = os.path.relpath(json_dir+"/"+basename+".json")
        picpath = os.path.relpath(jpg_dir+"/"+basename+".jpg")
        pngpath = os.path.relpath(png_dir+"/"+basename+".png")
        save_image(generated.detach().cpu()[0],png_path,normalize=True)
        if cnts is not None: #len(cnts) > 0:
            polygon = pp.save_contour(cnts,unc,tp,jsonpath,args)
            img_cv = pp.draw_pic(tp,polygon,args)
            img_cat = cv.vconcat([img_cat, img_cv])
            pp.save_pic(img_cat,picpath,args)

        result.append([pathA, unc, tp, error.item()])
        del real_A, encoded, generated, pathA, error
        torch.cuda.empty_cache()
    
   
    result.sort(reverse=True, key=lambda list: list[1])
    
    with open(tmp_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    

    
    
    