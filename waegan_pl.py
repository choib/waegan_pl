import torch
import datetime
import time
import sys
import logging
#import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image
import torchvision.transforms as transforms
from argparse import ArgumentParser, Namespace

#from models_resnet_pl import *
from models_pl import *
from datasets_resnet import *

import torch.distributed as dist
#from parallel import DataParallelModel as DPM 
#from parallel import DataParallelCriterion as DPC
from apex import *
import pytorch_ssim
import options
import save_im as sv
# import print_sysout as ps
# import eval_data_gt as edg
# import eval_data as ed
# import update_data as ud
import post_process as pp
import pandas as pd
import csv
import cv2 as cv

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import OrderedDict
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, rank_zero_only, rank_zero_warn
import os
#os.environ["NCCL_DEBUG"] = "INFO"

cuda = True if torch.cuda.is_available() else False
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print("GPU status: %d"%torch.cuda.device_count())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

#dataroot = "../data/ICT_10_26_mix_dataset"

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

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

class WaeGAN(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = args.n_z
        self.lr = args.lr
        self.n_critic = args.n_critic
        self.args = args
        self.smth = 0.45#args.smth
        #self.automatic_optimization=False
        #self.b1 = b1
        #self.b2 = b2
        self.batch_size = args.batch_size
        self.one = torch.tensor(1,dtype=torch.float)#.to(self.device)
        # if args.precision==16:
        #     self.one = self.one.half()
        # else:
        #     pass
        # self.mone = -1*self.one
        
        
        self.df_csv = f"./csv/{args.date}_{args.dataset}.csv"
        self.tmp_csv = f"./tmp/{args.date}_{args.dataset}_result.csv"
        self.tmp_pred = f"./tmp/{args.date}_{args.dataset}_predict.csv"
        # networks
        self.generator_unet = ResNetUNet(args)#.to(self.device)
        self.discriminator_unet = MultiDiscriminator(args)#.to(self.device)
        self.mse_loss = nn.MSELoss()#.to(self.device)
        self.adv_loss = torch.nn.BCEWithLogitsLoss()#.to(self.device)
        self.aux_loss = LabelSmoothingCrossEntropy(0.1)#LabelSmoothing(self.smth)#torch.nn.CrossEntropyLoss(label_smoothing=self.smth)# 
        self.criterion = pytorch_ssim.SSIM()#.to(self.device)

        self.generator_unet.apply(weights_init_normal)
        self.discriminator_unet.apply(weights_init_normal)

        self.k_enc = args.gp_lambda*args.k_wass
        self.no_sample = 0
        self.sum_test = 0
        self.json_dir = f"./tmp/{args.date}_{args.dataset}_json"
        self.jpg_dir = f"./tmp/{args.date}_{args.dataset}_jpg"
        self.png_dir = f"./tmp/{args.date}_{args.dataset}_png"
        self.org_dir = f"./tmp/{args.date}_{args.dataset}_org"
        self.gt_dir = f"./tmp/{args.date}_{args.dataset}_gt"
        self.pred_dir = f"./tmp/{args.date}_{args.dataset}_pred"
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.jpg_dir, exist_ok=True)
        os.makedirs(self.png_dir, exist_ok=True)
        os.makedirs(self.org_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        #clean_dir(self.json_dir)
        #clean_dir(self.jpg_dir)
        self.result =  []
        self.inv_transform = transforms.Compose(
            [
                UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.ToPILImage(),
            ]
        )

    def forward(self, z):
        return self.generator_unet(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator_unet.compute_out(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        lambda_gp = self.args.gp_lambda
        real_A = Variable(batch["A"],requires_grad=True)#.to(self.device)
        real_B = Variable(batch["B"],requires_grad=True)#.to(self.device)
        aug_A = Variable(batch["aug_A"],requires_grad=True)#.to(self.device)
        labels = Variable(LongTensor(batch["label"]), requires_grad=False)
            
        gen_labels = Variable(LongTensor(np.random.randint(0, self.args.n_classes,real_A.shape[0])),requires_grad=False)


        if self.args.noise_in:
            noisy = sv.gaussian(real_A,mean=0,stddev=self.args.sigma)#.to(self.device)
        else:
            noisy = aug_A
      
        if optimizer_idx == 0:
            frozen_params(self.discriminator_unet)
            free_params(self.generator_unet)

            generated, encoded_, e1_, _ = self(real_A)
            _, z_, z1_, _ = self(noisy)
            
            self.target, _ = torch.mode(torch.argmax(e1_, dim=1))
            match = (torch.argmax(e1_, dim=1) == self.target).type(Tensor)
            label_loss = self.aux_loss(z1_,e1_)
            self.log("matching",match.mean().item())
            self.log("label loss",label_loss)
            m_loss = self.mse_loss(real_B, generated)
            if self.args.descending:
                s_r = (1-self.current_epoch/self.args.train_max)*self.args.style_ratio
            else:
                s_r = (self.current_epoch/self.args.train_max)*self.args.style_ratio

            style_loss = (s_r)*(1 - self.criterion(real_B, generated))\
                + (1-s_r)* m_loss
            
           
            
            enc_loss =(self.mse_loss(encoded_ , z_)) 
            self.log("enc loss",enc_loss) 
            enc_loss = self.args.k_wass*enc_loss       

            h_loss = self.args.k_wass*self.discriminator_unet(generated)
            wass_loss = -torch.mean(h_loss)
            g_loss = (style_loss + wass_loss + enc_loss) 
           
            self.log("style loss",style_loss)
            self.log("mse loss",m_loss)
            self.log("g_loss",g_loss, sync_dist=True)
            self.log("wass loss",wass_loss)
            g_loss += self.args.k_wass*label_loss
            g_loss = g_loss.float()
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 1:
            free_params(self.discriminator_unet)
            free_params(self.generator_unet)
            valid = Variable(Tensor(real_A.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real_A.shape[0], 1).fill_(0.0), requires_grad=False)

            noisy = sv.gaussian(real_A,mean=0,stddev=self.args.sigma)
            generated, encoded_, e1_, e2_ = self(real_A.detach())
            _, z_, z1_, z2_ = self(noisy)

            real_aux, real_adv = e1_, e2_
            labels_onehot= torch.nn.functional.one_hot( labels, num_classes=self.args.n_classes)
            real_loss = self.adv_loss(real_adv,valid) + self.aux_loss(real_aux, labels_onehot)
           
            fake_aux, fake_adv = z1_, z2_
            gen_labels_onehot= torch.nn.functional.one_hot(gen_labels, num_classes=self.args.n_classes)
            fake_loss = self.adv_loss(fake_adv,fake) + self.aux_loss(fake_aux, gen_labels_onehot)
            
          
            enc_loss =(self.mse_loss(encoded_ , z_))
            self.log("enc loss",enc_loss) # just monitor
            gen_loss = self.args.k_wass*(real_loss + fake_loss)/4.0
            self.log("genenc loss",gen_loss) 
            f_loss = self.args.k_wass*self.discriminator_unet(real_B)
            h_loss = self.args.k_wass*self.discriminator_unet(generated)
            d_loss = (torch.mean(f_loss) - torch.mean(h_loss))#wasserstein loss
           
            if self.args.clip_weight:
                d_loss -= gen_loss#enc_loss# if self.args.gram else 0
                for p in self.discriminator_unet.parameters():
                    p.data.clamp_(-self.args.clip_value, self.args.clip_value)
            else:
                gradient_penalty = self.compute_gradient_penalty(real_B.data, generated.data)
                d_loss -= gen_loss#enc_loss# if self.args.gram else 0
                d_loss -= self.args.gp_lambda* self.args.k_wass* gradient_penalty
            
           
            d_loss = -d_loss.float()
            self.log("discriminator loss",d_loss, sync_dist=True)
            
            tqdm_dict = {'d_loss': d_loss}
            
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    # def backward(self, loss, optimizer, optimizer_idx):
    #     # do a custom way of backward
    #     with autocast():
    #         one = torch.tensor(1, dtype=torch.float) 
    #     if optimizer_idx == 0:
    #         with autocast():
    #             loss.backward(one,retain_graph=True)#.to(self.device)
    #     else:
    #         with autocast():
    #             loss.backward(-1*one,retain_graph=True)#.to(self.device)


    def configure_optimizers(self):
        
        lr = self.lr
        #b1 = self.b1
        #b2 = self.b2
        opt_g = torch.optim.Adam(self.generator_unet.parameters(), lr=lr)#, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator_unet.parameters(), lr=lr)#, betas=(b1, b2))
        
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': self.n_critic},
        )

    def train_dataloader(self):
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        dataset = ImageDataset("%s/%s" % (self.args.dataroot,self.args.dataset) , input_shape, mode='train')
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=24, pin_memory=True)

    def test_dataloader(self):
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        mode = self.args.val_target
        dataset = ImageDataset("%s/%s" % (self.args.dataroot,self.args.dataset), input_shape, mode=mode )
        return DataLoader(dataset, batch_size= self.args.test_batch_size, shuffle=False, num_workers=24)

    def predict_dataloader(self):
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        #dataset = UserDataset("../data/%s" % self.args.dataset, input_shape, mode="test")
        dataset = UserDataset("../data/temporary", input_shape, mode="test")
        #dataset = UserDataset("../data/SNUH_test0", input_shape, mode="test")
        return DataLoader(dataset, batch_size=self.args.test_batch_size, num_workers=16)
        
    def on_epoch_end(self):
        pass
    
    def on_fit_start(self) -> None:
        pl.seed_everything(42)
        return super().on_fit_start()

    def test_step(self, batch, batch_idx):
        for img_A, img_B, aug_A, pathA, pathB in zip(batch["A"], batch["B"], batch["aug_A"], batch["pathA"], batch["pathB"]):
            self.no_sample += 1
            real_A = Variable(img_A).type(Tensor)#.cuda()
            real_A = real_A.unsqueeze(0)
            real_B = Variable(img_B).type(Tensor)#.cuda()
            real_B = real_B.unsqueeze(0)
            aug_A = Variable(aug_A).type(Tensor)#.cuda()
            aug_A = aug_A.unsqueeze(0)
            fake_B, e, e1, e2 = self(real_A)
            _, z, z1, z2 = self(aug_A)
            nz_f = self.mse_loss(e, z) #+ self.mse_loss(e1, z1) + self.mse_loss(e2, z2)
            nz_f = nz_f.item()
            val = e2.squeeze().item()
            test_loss = self.mse_loss(fake_B, real_B)
            values, indexes = torch.topk(e1, k=3, dim=-1)
            indexes = indexes.data.view(-1)
            mean_v = values.mean()
            values = values - mean_v
            values = values.data.view(-1)
           
            real_A = real_A.data[0]
            fake_B = fake_B.data[0]
            real_B = real_B.data[0] #torch.cat([x for x in real_B.data.cpu()], -1)
            
            img_org = self.inv_transform(real_A.detach().clone())
            img_org = np.asarray(img_org, dtype='uint8')
            
            img_seg = self.inv_transform(fake_B.detach().clone())
            img_seg = np.asarray(img_seg, dtype='uint8')
           
            img_gt = self.inv_transform(real_B.detach().clone())#image_B
            img_gt = np.asarray(img_gt, dtype='uint8')
            
            target, t_area, area_p, _, _ =pp.critic_segmentation(img_seg)
            tp = target if area_p > 0.05 else 0
            #area_pred = t_area*area_p
            
            #tp = self.args.n_class
            iou, iou_bb, dice, unc, area_int, area_pred, area_gt, cnts = pp.critic_segmentation_by_class(tp, img_seg, img_gt, self.args)
            pngpath = os.path.relpath(self.org_dir+f"/{batch_idx}_{self.no_sample}.png")
            pp.save_pic(img_org,pngpath,0)
            pngpath = os.path.relpath(self.gt_dir+f"/{batch_idx}_{self.no_sample}.png")
            pp.save_pic(img_gt,pngpath,0)
            pngpath = os.path.relpath(self.pred_dir+f"/{batch_idx}_{self.no_sample}.png")
            pp.save_pic(img_seg,pngpath,0)          
            self.result.append([nz_f,tp, indexes[0].item(), indexes[1].item(), indexes[2].item(),values[0].item(), test_loss.item(), val, unc,  area_int, area_pred, area_gt, iou_bb, dice, iou, area_p,pathA, pathB])
            self.sum_test += test_loss.item()
            
            #del real_A, real_B, fake_B, pathA, pathB, test_loss, z, aug_A, cnts, nz_f, e, e1, e2, z1, z2
            #torch.cuda.empty_cache()
        self.log("test loss",self.sum_test/self.no_sample, sync_dist=True)
            

        tqdm_dict = {'test_loss': test_loss}
        
        output = OrderedDict({
            'loss': test_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output
   
    def predict_step(self, batch, batch_idx):
        for img_A, aug_A, pathA in zip(batch["A"], batch["aug_A"], batch["pathA"]):
            self.no_sample += 1
            real_A = Variable(img_A).type(Tensor)#.cuda()
            real_A = real_A.unsqueeze(0)
            aug_A = Variable(aug_A).type(Tensor)#.cuda()
            aug_A = aug_A.unsqueeze(0)
            fake_B, e, e1, e2 = self(real_A)
            _, z, z1, z2 = self(aug_A)
            nz_f = self.mse_loss(e, z) #+ self.mse_loss(e1, z1) + self.mse_loss(e2, z2)
            nz_f = nz_f.item()
           
            fake_B = fake_B.data[0]  
            real_A = real_A.data[0] 
            img_seg = self.inv_transform(fake_B.detach().clone())
            img_seg = np.asarray(img_seg, dtype='uint8')
            img_org = self.inv_transform(real_A.detach().clone())
            img_org = np.asarray(img_org, dtype='uint8')  
            #img_cat = cv.vconcat([img_seg, img_gt])
            
            #tp = self.args.n_class
            #tp = int(torch.argmax(e1.squeeze()).item())
            val = e2.squeeze().item()
            values, indexes = torch.topk(e1, k=3, dim=-1)
            indexes = indexes.data.view(-1)
            mean_v = values.mean()
            values = values - mean_v
            values = values.data.view(-1)
            target, area, area_p, cnts, clist =pp.critic_segmentation(img_seg)  
            base = os.path.splitext(pathA)[0]
            _, basename = os.path.split(base)
            jsonpath = os.path.relpath(self.json_dir+"/"+basename+".json")
            picpath = os.path.relpath(self.jpg_dir+"/"+basename+".jpg")
            pngpath = os.path.relpath(self.png_dir+"/"+basename+".png")
            #pp.save_pic(img_seg,picpath,0)
            pp.save_pic(img_org+img_seg,picpath,0)
            if cnts is not None: #len(cnts) > 0:
                M = cv.moments(cnts[0])
                M0 = M['m00']
                cX = int(M['m10'] / M0) if M0 > 0 else 0
                cY = int(M['m01'] / M0) if M0 > 0 else 0
                polygon = pp.save_contour(cnts,area_p,target,jsonpath,self.args)
                img_cv = pp.draw_pic(target,polygon,self.args)
                #img_cat = cv.vconcat([img_cat, img_cv])
                pp.save_pic(img_cv,pngpath,0)
            else:
                cX = cY = 0
            self.result.append([target, indexes[0].item(), indexes[1].item(), indexes[2].item(),values[0].item(),val, nz_f, clist, area, area_p, cX, cY, pathA])

        tqdm_dict = {'predict_loss': nz_f}
        
        output = OrderedDict({
            'loss': nz_f,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    @rank_zero_only
    def on_predict_end(self):
        #dist.all_gather(self.result, Tensor)
        #output = [None for _ in self.result]
       
        self.result.sort(reverse=False, key=lambda list: list[5])
        sys.stdout.write("Sorted!!\n")
        logging.info("Sorted!!\n")

        with open(self.tmp_pred,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.result)

    @rank_zero_only
    def on_test_end(self):
        mean_test = self.sum_test / self.no_sample
        str = "mean mse: {}\n".format(mean_test)
        sys.stdout.write(str)
        logging.info(str)

        self.result.sort(reverse=True, key=lambda list: list[0])
        sys.stdout.write("Sorted!!\n")
        logging.info("Sorted!!\n")

        with open(self.tmp_csv,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.result)

        world = args.img_width * args.img_height
        #[nz_f,tp, indexes[0].item(), indexes[1].item(), indexes[2].item(),values[0].item(), test_loss.item(), pathA, pathB, unc,  area_int, area_pred, area_gt, iou_bb, dice, iou]
        df = pd.read_csv(self.tmp_csv, names=['nz_f', 'class', 'pclass', '2nd','3rd', 'diff','loss','validity','uncertainty',\
            'intersection','pred','gt','iou bb','f1','iou','area_p','pathA','pathB'], header=None) 

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
        str = "Count={}, F1={},TP={},TN={},FP={},FN={},\nACC={},IoU={},bbIoU={},mAUC={}, nz_f={} {}, validity={} {}, area_p={}\n".format(len(df), df['f1'].mean(), df['TP'].mean(),\
                df['TN'].mean(),df['FP'].mean(),df['FN'].mean(),df['ACC'].mean(),df['iou'].mean(),df['iou bb'].mean(),df['mAUC'].mean(),df['nz_f'].quantile(0.5),df['nz_f'].std(),df['validity'].mean(),df['validity'].std(),df['area_p'].mean())
        sys.stdout.write(str)
        logging.info(str)
      
        top5 = df['pclass'].value_counts().head()
        str = f"1st top5: index counts\n{top5}\n"
        sys.stdout.write(str)
        logging.info(str)
        top5 = df['2nd'].value_counts().head()
        str = f"2nd top5: index counts\n{top5}\n"
        sys.stdout.write(str)
        logging.info(str)
        top5 = df['3rd'].value_counts().head()
        str = f"3rd top5: index counts\n{top5}\n"
        sys.stdout.write(str)
        logging.info(str)

        df.to_csv(self.df_csv,index=False)

class SaveImage(Callback):
    def __init__(self,args) -> None:
        self.args = args
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        self.dataset = ImageDataset("%s/%s" % (self.args.dataroot,self.args.dataset), input_shape, mode='test')
        self.test_loader = DataLoader(self.dataset, batch_size= self.args.test_batch_size, shuffle=True, num_workers=16)
       
    #@rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        batches_done = pl_module.current_epoch
        pl_module.generator_unet.eval()
        val_loss, validity = sv.sample_images(batches_done, self.test_loader, self.args, pl_module.generator_unet, pl_module.mse_loss, Tensor)
        self.log("validation loss",val_loss)
        self.log("validity",validity)
        pl_module.generator_unet.train()


def main(args: Namespace) -> None:
    input_shape = (args.n_channel, args.img_height, args.img_width)
    if args.precision == 16:
        Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    o_level = 'O1'
    amp_back = 'apex'
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WaeGAN(args)
    if amp_back == 'apex':
        model = amp.initialize(model.cuda(), opt_level=o_level,loss_scale=1.0)
        amp.state_dict()
    dataset = args.dataset
    date = args.date
    save_path = "./save/{dataset}_{date}".format(dataset=dataset,date=date)
    checkpoint_callback = ModelCheckpoint(monitor="mse loss", dirpath=save_path,
        filename="waegan-{epoch:02d}",
        save_top_k=3,
        mode="min",
        save_last=True)
    saveim_callback = SaveImage(args)
    precision = args.precision
    accel = "ddp" if args.DDP else None
    callbacks = [checkpoint_callback,saveim_callback]
    logging.basicConfig(filename="./%s.log" % args.date ,format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if args.epoch !=0:
        # Load pretrained models
        start_epoch = args.epoch - 1
        if args.last:
            ckpt = ModelCheckpoint(dirpath=save_path,filename="last")
        else:
            ckpt = ModelCheckpoint(dirpath=save_path,filename="waegan-{epoch:02d}")
           
        #ckpt = ModelCheckpoint(dirpath=save_path,filename="waegan-{epoch:02d}")
        base = os.path.basename(ckpt.format_checkpoint_name(dict(epoch=start_epoch)))
        ckpt_path = os.path.join(save_path,base)
    trainer = Trainer(gpus=args.gpu,accelerator=accel,callbacks=callbacks,\
            precision=precision, amp_level= o_level, amp_backend=amp_back,\
            log_every_n_steps=10, auto_select_gpus=True, max_epochs= args.train_max,\
            auto_scale_batch_size="binsearch", accumulate_grad_batches=1,
            sync_batchnorm=True)#gradient_clip_val=args.gp_lambda,
    
    if args.train:
        #trainer.tune(model)
        if args.epoch !=0:
            model = model.load_from_checkpoint(ckpt_path)
            model.train()
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)
    else:
       
        if args.val_target=='train':
            model = model.load_from_checkpoint(ckpt_path)
            model.eval()
            input_shape = (args.n_channel, args.img_height, args.img_width)
            dataset = ImageDataset("%s/%s" % (args.dataroot,args.dataset) , input_shape, mode='train')
            train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=24)
            trainer.test(model,dataloaders=train_dataloader)
            
    
        elif args.val_target=='test':

            model = model.load_from_checkpoint(ckpt_path)
            model.eval()
            
            trainer.test(model)
            
        elif args.val_target=='user':

            model = model.load_from_checkpoint(ckpt_path)
            model.eval()
            
            trainer.predict(model)
      
        else:
            print("Nothing to be done !")

if __name__ == '__main__':
    
    args = options.Options()
    args = options.Options.parse(args)
    sv.init_imdirs(args)
    main(args)




