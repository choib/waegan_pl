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

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
        

def eval_data_gt(args,generator_unet, data_loader, Tensor, criterion):

    generator_unet.eval()
    input_shape = (args.n_channel, args.img_height, args.img_width)
    sum_test = 0
    tmp_path = "./tmp/img_cv.jpg"
    tmp_csv = "./tmp/result.csv"
    json_dir = "./tmp/json"
    jpg_dir = "./tmp/jpg"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    clean_dir(json_dir)
    clean_dir(jpg_dir)
    result = []

    for i, batch in enumerate(data_loader):
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        #encoded = wae_encoder(real_A)
        generated, _, encoded, _ = generator_unet(real_A)
        test_loss = criterion(generated, real_B)
        #sum_test += test_lossfake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        save_image(generated.detach().cpu()[0],tmp_path)
        img_cv = Image.open(tmp_path)
        img_cv.load()
        img_cv = img_cat = np.asarray(img_cv, dtype='uint8')
        tp, unc, area_p, cnts =pp.critic_segmentation(img_cv)
        tp = 2 if unc > args.uncertainty else tp

        img_cv = save_image(real_B.detach().cpu()[0],tmp_path)
        img_cv = Image.open(tmp_path)
        img_cv.load()
        img_cv = np.asarray(img_cv, dtype='uint8')
        img_cat = cv.vconcat([img_cat, img_cv])
        t, _, area_gt, _ = pp.critic_segmentation(img_cv)
        a_ratio= area_p/area_gt if area_gt > 0 else 0

        pathA = batch["pathA"]
        pathB = batch["pathB"]
        base = os.path.splitext(pathB[0])[0]
        _, basename = os.path.split(base)
        jsonpath = os.path.relpath(json_dir+"/"+basename+".json")
        picpath = os.path.relpath(jpg_dir+"/"+basename+".jpg")
        if cnts is not None: #and len(cnts) > 0:
            polygon = pp.save_contour(cnts,unc,tp,jsonpath,args)
            img_cv = pp.draw_pic(tp,polygon,args)
            img_cat = cv.vconcat([img_cat, img_cv])
            pp.save_pic(img_cat,picpath,args)

        result.append([test_loss.detach().cpu().numpy(), pathA, pathB, unc, tp, t, a_ratio])
        sum_test += test_loss.detach().cpu().numpy()
        #sys.stdout.write("\tFile A:{} \tTest Loss: {:<10.6e}\n".format("".join(pathA), test_loss.data.item()))
        #logging.info("File A:{} \tTest Loss: {:<10.6e}\n".format("".join(pathA), test_loss.data.item()))
        del real_A, real_B, encoded, generated, pathA, pathB, test_loss
        torch.cuda.empty_cache()
    
    mean_test = sum_test / len(data_loader)
    str = "mean mse: {}\n".format(mean_test)
    sys.stdout.write(str)
    logging.info(str)

    result.sort(reverse=True, key=lambda list: list[0])
    sys.stdout.write("Sorted!!\n")
    logging.info("Sorted!!\n")

    
    with open(tmp_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    df = pd.read_csv(tmp_csv, names=['loss','pathA','pathB','uncertainty','predict','target','iou'], header=None) 

    y_pred = df["predict"]
    y_actu = df["target"]

    #cm = pd.crosstab(y_actu,y_pred,rownames=['Actual'], colnames=['Predicted'], margins=True)
    cm = confusion_matrix(y_actu,y_pred)
    str = "confusion matrix \n{}\n".format(cm)
    sys.stdout.write(str)
    logging.info(str)    
    cm = multilabel_confusion_matrix(y_actu, y_pred)
    str = "multilabel confusion matrix \n{}\n".format(cm)
    sys.stdout.write(str)
    logging.info(str)

    for i in range(len(cm)):
        acc = (cm[i][0][0]+cm[i][1][1])/sum(map(sum,cm[i]))
        str="class {}: accuracy {}\n".format(i,acc)
        sys.stdout.write(str)
        logging.info(str)

    y_pred = df.loc[df["uncertainty"]<args.uncertainty,"predict"]
    y_actu = df.loc[df["uncertainty"]<args.uncertainty,"target"]

    #cm = pd.crosstab(y_actu,y_pred,rownames=['Actual'], colnames=['Predicted'], margins=True)
    cm = confusion_matrix(y_actu,y_pred)
    str = "confusion matrix wrt uncertainty\n{}\n".format(cm)
    sys.stdout.write(str)
    logging.info(str)

    # FP = cm.sum(axis=0) - np.diag(cm) 
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)
    # FP = FP.astype(float)
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy for each class
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    # str = "TP={},TN={},FP={},FN={},ACC={}\n".format(TP,TN,FP,FN,ACC)
    # sys.stdout.write(str)
    # logging.info(str)

    cm = multilabel_confusion_matrix(y_actu, y_pred)
    str = "multilabel confusion matrix wrt uncertainty\n{}\n".format(cm)
    sys.stdout.write(str)
    logging.info(str)

    for i in range(len(cm)):
        acc = (cm[i][0][0]+cm[i][1][1])/sum(map(sum,cm[i]))
        str="class {}: accuracy {}\n".format(i,acc)
        sys.stdout.write(str)
        logging.info(str)

    validation_transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    real_samples = None
    img_samples = None
    gt_samples = None
    enc_samples = None
    no_sample = 0    
    for i in range(20):
        str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {:<10.6e}\tclass: {}, {}\tratio:{}\n".format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][5], result[i][6])
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
        enc_samples = encoded if enc_samples is None else torch.cat((enc_samples,encoded),0)
        no_sample += 1
    
    for i in range(-1,-1):
        str = "{}\tFile A:{} \tTest Loss: {:<10.6e}\tuncertainty: {:<10.6e}\tclass: {}, {}\tratio:{}\n".format(i,result[i][1],result[i][0], result[i][3], result[i][4], result[i][5], result[i][6])
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
        #encoded = wae_encoder(real_A)
        fake_B, _, encoded, _ = generator_unet(real_A)
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        real_A = torch.cat([x for x in real_A.data.cpu()], -1)
        real_B = torch.cat([x for x in real_B.data.cpu()], -1)
      
        real_samples = real_A if real_samples is None else torch.cat((real_samples, real_A), 1)
        gt_samples = real_B if gt_samples is None else torch.cat((gt_samples, real_B), 1)
        img_samples = fake_B if img_samples is None else torch.cat((img_samples, fake_B), 1)
        enc_samples = encoded if enc_samples is None else torch.cat((enc_samples,encoded),0)
        no_sample += 1
    #real_samples = torch.cat((gt_samples, img_samples), 0)   
    save_image(real_samples, "images/%s_result_%s/%s_photo.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(gt_samples, "images/%s_result_%s/%s_gt.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(img_samples, "images/%s_result_%s/%s_seg.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
    save_image(enc_samples, "images/%s_result_%s/%s_enc.jpg" % (args.date, args.dataset, args.epoch), nrow=1, normalize=True)
       