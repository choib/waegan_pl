import post_process as pp
import pandas as pd
import csv
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
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
import re

reg_tok = re.compile("\['(.*)'\]")

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

def update_data(args,generator_unet, Tensor):

    generator_unet.eval()
    input_shape = (args.n_channel, args.img_height, args.img_width)
    sum_test = 0
    tmp_path = "./tmp/user_cv.jpg"
    tmp_csv = "./tmp/user_result.csv"
    json_dir = "./tmp/user_json"
    jpg_dir = "./tmp/user_jpg"
    png_dir = "./tmp/user_png"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    clean_dir(json_dir)
    clean_dir(jpg_dir)
    clean_dir(png_dir)

    df = pd.read_csv(tmp_csv, names=['pathA','uncertainty','target'], header=None) 

    validation_transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    for index, row in df.iterrows():
        pathA = reg_tok.findall(row["pathA"])[0]
        if row['target'] == -1:
            os.remove(os.path.relpath("".join(pathA)))
            continue
        img_cv = img_cat = Image.open(os.path.relpath("".join(pathA)))
        img_cat.load()
        img_cat = np.asarray(img_cat, dtype='uint8')

        item_A = validation_transform(img_cv)
        real_A = Variable(item_A.type(Tensor))
        real_A = real_A.unsqueeze(0)
        
        generated, _, encoded, _ = generator_unet(real_A)
        save_image(generated.detach().cpu()[0],tmp_path)
        img_cv = Image.open(tmp_path)
        img_cv.load()
        img_cv = np.asarray(img_cv, dtype='uint8')
        tt = row['target']
        unc = row['uncertainty']
        area_p, cnts =pp.update_segmentation(img_cv)
        img_cat = cv.vconcat([img_cat, img_cv])
        
        base = os.path.splitext(pathA)[0]
        _, basename = os.path.split(base)
        jsonpath = os.path.relpath(json_dir+"/"+basename+".json")
        picpath = os.path.relpath(jpg_dir+"/"+basename+".jpg")
        resultpath = os.path.relpath(png_dir+"/"+basename+".png")
        if cnts is not None:
            polygon = pp.save_contour(cnts,unc,tt,jsonpath,args)
            img_cv = pp.draw_pic(tt,polygon,args)
            pp.save_pic(img_cv,resultpath,args)
            img_cat = cv.vconcat([img_cat, img_cv])
            pp.save_pic(img_cat,picpath,args)

        del real_A, encoded, generated, pathA
        torch.cuda.empty_cache()
    

    

    
    
    