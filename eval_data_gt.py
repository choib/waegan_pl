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
import matplotlib.pyplot as plt
import math

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
        

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def eval_data_gt(args, generator_unet, data_loader, Tensor, criterion):

    #generator_unet.eval()
    input_shape = (args.n_channel, args.img_height, args.img_width)
    sum_test = 0
    no_sample = 0
    #tmp_seg_path = f"./tmp/{args.date}_{args.dataset}_seg.png"
    #tmp_gt_path = f"./tmp/{args.date}_{args.dataset}.png"
    df_csv = f"./csv/{args.date}_{args.dataset}.csv"
    tmp_csv = f"./tmp/{args.date}_{args.dataset}_result.csv"
    json_dir = f"./tmp/{args.date}_{args.dataset}_json"
    jpg_dir = f"./tmp/{args.date}_{args.dataset}_jpg"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    clean_dir(json_dir)
    clean_dir(jpg_dir)

    result = []
    iou_list=[]
    not_detected=[]
    #unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    validation_transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    inv_transform = transforms.Compose(
        [
            UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage(),
        ]
    )

    for i, batch in enumerate(data_loader):
        for img_A, img_B, aug_A, pathA, pathB in zip(batch["A"], batch["B"], batch["aug_A"], batch["pathA"], batch["pathB"]):
            no_sample += 1
            #real_A = img_A.unsqueeze(0)#.view(1, *img_A.shape)
            real_A = Variable(img_A).type(Tensor)#.cuda()
            real_A = real_A.unsqueeze(0)
            #real_B = img_B.unsqueeze(0)#.view(1, *img_B.shape)
            real_B = Variable(img_B).type(Tensor)#.cuda()
            real_B = real_B.unsqueeze(0)
            #aug_A = aug_A.unsqueeze(0)#.view(1, *aug_A.shape)
            aug_A = Variable(aug_A).type(Tensor)#.cuda()
            aug_A = aug_A.unsqueeze(0)
            fake_B, e, e1, e2 = generator_unet(real_A)
            _, z, z1, z2 = generator_unet(aug_A)
            nz_f = criterion(e, z) + criterion(e1, z1) + criterion(e2, z2)
            nz_f = nz_f.item()
            test_loss = criterion(fake_B, real_B)
            fake_B = fake_B.data[0]
            real_B = real_B.data[0] #torch.cat([x for x in real_B.data.cpu()], -1)
            
            
            img_seg = inv_transform(fake_B)
            img_seg = np.asarray(img_seg, dtype='uint8')
            #img_seg = cv.cvtColor(img_seg, cv.COLOR_BGR2RGB)
            #image_B = Image.open(os.path.relpath("".join(pathB)))
            img_gt = inv_transform(fake_B)#image_B
            img_gt = np.asarray(img_gt, dtype='uint8')
            
            img_cat = cv.vconcat([img_seg, img_gt])
            #t, _, area_gt, _ = pp.critic_segmentation(img_cv)
            tp = args.n_class
            iou, iou_bb, dice, unc, area_int, area_p, area_gt, cnts = pp.critic_segmentation_by_class(tp, img_seg, img_gt, args)
            a_ratio= area_p/area_gt if area_gt > 0 else 0

            # pathA = pathA
            # pathB = pathB
            base = os.path.splitext(pathB)[0]
            _, basename = os.path.split(base)
            jsonpath = os.path.relpath(json_dir+"/"+basename+".json")
            picpath = os.path.relpath(jpg_dir+"/"+basename+".jpg")
            if cnts is not None:#len(cnts) > 0:
                img_cvs = np.zeros(np.shape(img_gt),dtype=np.uint8)
                for i,cnt in enumerate(cnts):
                    polygon = pp.save_contour(cnt,unc,tp,jsonpath,args)
                    img_cv = pp.draw_pic(tp,polygon,args)
                    img_cvs = cv.add(img_cvs,img_cv)
                img_cat = cv.vconcat([img_cat, img_cvs])
                pp.save_pic(img_cat,picpath,0) # as it is
            result.append([test_loss.item(), pathA, pathB, unc, tp, area_int, area_p, area_gt, iou_bb, dice, iou, nz_f])
            sum_test += test_loss.item()
            # str= f"\tTest Loss: {test_loss.detach().cpu().numpy()}, F1 {dice}, intersection {area_int}\n"
            # sys.stdout.write(str)
            # logging.info(str)
            del real_A, real_B, fake_B, pathA, pathB, test_loss, z, aug_A, cnts, nz_f, e, e1, e2, z1, z2
            torch.cuda.empty_cache()
    mean_test = sum_test / no_sample
    str = "mean mse: {}\n".format(mean_test)
    sys.stdout.write(str)
    logging.info(str)

    result.sort(reverse=True, key=lambda list: list[0])
    sys.stdout.write("Sorted!!\n")
    logging.info("Sorted!!\n")

    
    with open(tmp_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    world = args.img_width * args.img_height
    df = pd.read_csv(tmp_csv, names=['loss','pathA','pathB','uncertainty','class',\
        'intersection','pred','gt','iou bb','f1','iou', 'nz f'], header=None) 

    df['intersection'].astype(float)
    df['pred'].astype(float)
    df['gt'].astype(float)
    
    df['FP'] = df['pred'].sub(df['intersection']) #Predict an event when there was no event.
    df['FN'] = df['gt'].sub(df['intersection']) #Predict no event when in fact there was an event.
    df['TP'] = df['intersection']
    df['TN'] = df['intersection'].add(world - df['pred'].add(df['gt']))

    df['FP'].astype(float)
    df['FN'].astype(float)
    df['TP'].astype(float)
    df['TN'].astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    df['TPR'] = df['TP'].div(df['TP'].add(df['FN']))
    # Specificity or true negative rate
    df['TNR'] = df['TN'].div(df['TN'].add(df['FP']))
    # Precision or positive predictive value
    df['PPV'] = df['TP'].div(df['TP'].add(df['FP']))
    # Negative predictive value
    df['NPV'] = df['TN'].div(df['TN'].add(df['FN']))
    # Fall out or false positive rate
    df['FPR'] = df['FP'].div(df['FP'].add(df['TN']))
    # False negative rate
    df['FNR'] = df['FN'].div(df['TP'].add(df['FN']))
    # False discovery rate
    df['FDR'] = df['FP'].div(df['TP'].add(df['FP']))
    # Overall accuracy for each class
    df['ACC'] = (df['TP'].add(df['TN'])).div((df['TP'].add(df['FP'].add(df['FN'].add(df['TN'])))))
    # Approx AUC
    df['mAUC'] = 0.5*(df['TPR'].add(df['FPR']))
    str = "Count={}, F1={},TP={},TN={},FP={},FN={},ACC={},IoU={},bbIoU={},mAUC={}\n".format(len(df), df['f1'].mean(), df['TP'].mean(),\
            df['TN'].mean(),df['FP'].mean(),df['FN'].mean(),df['ACC'].mean(),df['iou'].mean(),df['iou bb'].mean(),df['mAUC'].mean())
    sys.stdout.write(str)
    logging.info(str)

    # for i in range(20):
    #     str = "TP={},TN={},FP={},FN={},ACC={}\n".format(df['TP'].iloc[i],\
    #         df['TN'].iloc[i],df['FP'].iloc[i],df['FN'].iloc[i],df['ACC'].iloc[i])
    #     sys.stdout.write(str)
    #     logging.info(str)

    df.to_csv(df_csv,index=False)

    real_samples = None
    img_samples = None
    gt_samples = None
    enc_samples = None
    no_sample = 0    
    for i in range(20):
        # str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {:<10.6e}\tclass: {}\tf1: {}\n"\
        #     .format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][9])
        str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {}\tclass: {}\tf1: {}\n"\
            .format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][9])
        sys.stdout.write(str)
        logging.info(str)
        image_A = Image.open(os.path.relpath("".join(result[i][1])))
        image_B = Image.open(os.path.relpath("".join(result[i][2])))
        item_A = validation_transform(image_A)
        item_B = validation_transform(image_B)
        real_A = Variable(item_A.type(Tensor))
        real_A = real_A.unsqueeze(0)
        real_B = Variable(item_B.type(Tensor))
        real_B = real_B.unsqueeze(0)
        
        fake_B, _, encoded, _ = generator_unet(real_A)
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        real_A = torch.cat([x for x in real_A.data.cpu()], -1)
        real_B = torch.cat([x for x in real_B.data.cpu()], -1)
      
        real_samples = real_A if real_samples is None else torch.cat((real_samples, real_A), 1)
        gt_samples = real_B if gt_samples is None else torch.cat((gt_samples, real_B), 1)
        img_samples = fake_B if img_samples is None else torch.cat((img_samples, fake_B), 1)
        enc_samples = encoded if enc_samples is None else torch.cat((enc_samples,encoded),1)
        no_sample += 1
    
    save_image(real_samples, "images/%s_result_%s/%s_photo.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(gt_samples, "images/%s_result_%s/%s_gt.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(img_samples, "images/%s_result_%s/%s_seg.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(enc_samples, "images/%s_result_%s/%s_enc.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
       