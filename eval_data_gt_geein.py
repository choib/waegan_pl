import post_process as pp
import pandas as pd
import csv
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from pathlib import Path
import torch
import logging
import torch.nn.functional as F
from torchvision.utils import save_image
from datasets import *
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
        


def eval_data_gt(args, generator_unet, data_loader, Tensor, criterion):

    #generator_unet.eval()
    input_shape = (args.n_channel, args.img_height, args.img_width)
    sum_test = 0
    tmp_seg_path = "./tmp/img_seg.png"
    tmp_gt_path = "./tmp/img_gt.png"
    df_csv = f"./csv/{args.date}_{args.dataset}.csv"
    tmp_csv = "./tmp/result.csv"
    json_dir = "./tmp/json"
    jpg_dir = "./tmp/jpg"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    clean_dir(json_dir)
    clean_dir(jpg_dir)
    result = []
    iou_list=[]
    not_detected=[]
    running_accuracy=0
    validation_transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    for i, batch in enumerate(data_loader):
        real_A = Variable(batch["A"]).type(Tensor)#.cuda()a
        target = Variable(batch["target"]).type(Tensor)
        #real_B = Variable(batch["B"]).type(Tensor)#.cuda()
        aug_A = Variable(batch["aug_A"]).type(Tensor)#.cuda()
        fake_B, e, e1, e2 = generator_unet(real_A)
        _, z, z1, z2 = generator_unet(aug_A)
        # print(e,e1,e2)        
        output = F.softmax(e1, dim=1)
        pred_v, pred_idx = torch.topk(output, k=1)

        pred_v = pred_v.detach().cpu().numpy()
        pred_idx = pred_idx.detach().cpu().numpy()
        print(pred_idx)
        print(pred_v)
        print(target)
        print("############ljouhji#####")
        nz_f = criterion(e, z) + criterion(e1, z1) + criterion(e2, z2)
        nz_f = nz_f.item()
        print('########rrrrr#########')
        #print(nz_f)
        correct = (torch.argmax(output, dim=1) == target.squeeze()).type(torch.FloatTensor)
        running_accuracy += correct.mean()
    current_accuracy= running_accuracy / len(data_loader)
    print("accuracy is ")
    print(current_accuracy)
        #fake_B = fake_B.data.cpu()[0]
        #real_B = real_B.data.cpu()[0] #torch.cat([x for x in real_B.data.cpu()], -1)
        #test_loss = criterion(fake_B, real_B)
        #sum_test += test_lossfake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        #fake_B = cv.cvtColor(fake_B, cv.COLOR_BGR2RGB)
        #img_cv = fake_B.data.numpy() * 255
        #save_image(fake_B,tmp_seg_path,normalize=True)
        #img_cv = Image.open(tmp_seg_path)
        #img_cv.load()
        #img_seg = np.asarray(img_cv, dtype='uint8')
        #img_seg = cv.cvtColor(img_seg, cv.COLOR_BGR2RGB)
        #tp, unc, area_p, cnts =pp.critic_segmentation(img_cv)
        #real_B = cv.cvtColor(real_B, cv.COLOR_BGR2RGB)
        #save_image(real_B,tmp_gt_path,normalize=True)
        #img_cv = Image.open(tmp_gt_path)
        #img_cv.load()
        #img_cv = real_B.data.numpy() * 255
    #     img_gt = np.asarray(img_cv, dtype='uint8')
    #     #img_gt = cv.cvtColor(img_gt, cv.COLOR_BGR2RGB)
    #     img_cat = cv.vconcat([img_seg, img_gt])
    #     #t, _, area_gt, _ = pp.critic_segmentation(img_cv)
    #     tp = args.n_class
    #     iou, iou_bb, dice, unc, area_int, area_p, area_gt, cnts = pp.critic_segmentation_by_class(tp, img_seg, img_gt, args)
    #     a_ratio= area_p/area_gt if area_gt > 0 else 0

    #     pathA = batch["pathA"]
    #     pathB = batch["pathB"]
    #     base = os.path.splitext(pathB[0])[0]
    #     _, basename = os.path.split(base)
    #     jsonpath = os.path.relpath(json_dir+"/"+basename+".json")
    #     picpath = os.path.relpath(jpg_dir+"/"+basename+".jpg")
    #     if cnts is not None:#len(cnts) > 0:
    #         img_cvs = np.zeros(np.shape(img_cv),dtype=np.uint8)
    #         for i,cnt in enumerate(cnts):
    #             polygon = pp.save_contour(cnt,unc,tp,jsonpath,args)
    #             img_cv = pp.draw_pic(tp,polygon,args)
    #             img_cvs = cv.add(img_cvs,img_cv)
    #         img_cat = cv.vconcat([img_cat, img_cvs])
    #         pp.save_pic(img_cat,picpath,args)

    #     result.append([test_loss.item(), pathA, pathB, unc, tp, area_int, area_p, area_gt, iou_bb, dice, iou, nz_f])
    #     sum_test += test_loss.item()
    #     # str= f"\tTest Loss: {test_loss.detach().cpu().numpy()}, F1 {dice}, intersection {area_int}\n"
    #     # sys.stdout.write(str)
    #     # logging.info(str)
    #     del real_A, real_B, fake_B, pathA, pathB, test_loss, z, aug_A, cnts, nz_f, e, e1, e2, z1, z2
    #     torch.cuda.empty_cache()
    # mean_test = sum_test / len(data_loader)
    # str = "mean mse: {}\n".format(mean_test)
    # sys.stdout.write(str)
    # logging.info(str)

    # result.sort(reverse=True, key=lambda list: list[0])
    # sys.stdout.write("Sorted!!\n")
    # logging.info("Sorted!!\n")

    
    # with open(tmp_csv,"w",newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(result)

    # world = args.img_width * args.img_height
    # df = pd.read_csv(tmp_csv, names=['loss','pathA','pathB','uncertainty','class',\
    #     'intersection','pred','gt','iou bb','f1','iou', 'nz f'], header=None) 

    # df['intersection'].astype(float)
    # df['pred'].astype(float)
    # df['gt'].astype(float)
    
    # df['FP'] = df['pred'].sub(df['intersection']) #Predict an event when there was no event.
    # df['FN'] = df['gt'].sub(df['intersection']) #Predict no event when in fact there was an event.
    # df['TP'] = df['intersection']
    # df['TN'] = df['intersection'].add(world - df['pred'].add(df['gt']))

    # df['FP'].astype(float)
    # df['FN'].astype(float)
    # df['TP'].astype(float)
    # df['TN'].astype(float)

    # # Sensitivity, hit rate, recall, or true positive rate
    # df['TPR'] = df['TP'].div(df['TP'].add(df['FN']))
    # # Specificity or true negative rate
    # df['TNR'] = df['TN'].div(df['TN'].add(df['FP']))
    # # Precision or positive predictive value
    # df['PPV'] = df['TP'].div(df['TP'].add(df['FP']))
    # # Negative predictive value
    # df['NPV'] = df['TN'].div(df['TN'].add(df['FN']))
    # # Fall out or false positive rate
    # df['FPR'] = df['FP'].div(df['FP'].add(df['TN']))
    # # False negative rate
    # df['FNR'] = df['FN'].div(df['TP'].add(df['FN']))
    # # False discovery rate
    # df['FDR'] = df['FP'].div(df['TP'].add(df['FP']))
    # # Overall accuracy for each class
    # df['ACC'] = (df['TP'].add(df['TN'])).div((df['TP'].add(df['FP'].add(df['FN'].add(df['TN'])))))
    # # Approx AUC
    # df['mAUC'] = 0.5*(df['TPR'].add(df['FPR']))
    # str = "Count={}, F1={},TP={},TN={},FP={},FN={},ACC={},IoU={},bbIoU={},mAUC={}\n".format(len(df), df['f1'].mean(), df['TP'].mean(),\
    #         df['TN'].mean(),df['FP'].mean(),df['FN'].mean(),df['ACC'].mean(),df['iou'].mean(),df['iou bb'].mean(),df['mAUC'].mean())
    # sys.stdout.write(str)
    # logging.info(str)

    # # for i in range(20):
    # #     str = "TP={},TN={},FP={},FN={},ACC={}\n".format(df['TP'].iloc[i],\
    # #         df['TN'].iloc[i],df['FP'].iloc[i],df['FN'].iloc[i],df['ACC'].iloc[i])
    # #     sys.stdout.write(str)
    # #     logging.info(str)

    # df.to_csv(df_csv,index=False)

    
    # real_samples = None
    # img_samples = None
    # gt_samples = None
    # enc_samples = None
    # no_sample = 0    
    # for i in range(20):
    #     # str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {:<10.6e}\tclass: {}\tf1: {}\n"\
    #     #     .format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][9])
    #     str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {}\tclass: {}\tf1: {}\n"\
    #         .format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][9])
    #     sys.stdout.write(str)
    #     logging.info(str)
    #     image_A = Image.open(os.path.relpath("".join(result[i][1])))
    #     image_B = Image.open(os.path.relpath("".join(result[i][2])))
    #     item_A = validation_transform(image_A)
    #     item_B = validation_transform(image_B)
    #     real_A = Variable(item_A.type(Tensor))
    #     real_A = real_A.unsqueeze(0)
    #     real_B = Variable(item_B.type(Tensor))
    #     real_B = real_B.unsqueeze(0)
        
    #     fake_B, _, encoded, _ = generator_unet(real_A)
    #     fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
    #     real_A = torch.cat([x for x in real_A.data.cpu()], -1)
    #     real_B = torch.cat([x for x in real_B.data.cpu()], -1)
      
    #     real_samples = real_A if real_samples is None else torch.cat((real_samples, real_A), 1)
    #     gt_samples = real_B if gt_samples is None else torch.cat((gt_samples, real_B), 1)
    #     img_samples = fake_B if img_samples is None else torch.cat((img_samples, fake_B), 1)
    #     enc_samples = encoded if enc_samples is None else torch.cat((enc_samples,encoded),1)
    #     no_sample += 1
    
    # save_image(real_samples, "images/%s_result_%s/%s_photo.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    # save_image(gt_samples, "images/%s_result_%s/%s_gt.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    # save_image(img_samples, "images/%s_result_%s/%s_seg.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    # save_image(enc_samples, "images/%s_result_%s/%s_enc.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
       
