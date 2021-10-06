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

from models_resnet_pl import *
from datasets_resnet import *

#import torch.distributed as dist
#from parallel import DataParallelModel as DPM 
#from parallel import DataParallelCriterion as DPC

import pytorch_ssim
import options
import save_im as sv
import print_sysout as ps
import eval_data_gt as edg
import eval_data as ed
import update_data as ud

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import OrderedDict
from torch.cuda.amp import autocast



cuda = True if torch.cuda.is_available() else False
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print("GPU status: %d"%torch.cuda.device_count())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class WaeGAN(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = args.n_z
        self.lr = args.lr
        self.n_critic = args.n_critic
        #self.automatic_optimization=False
        #self.b1 = b1
        #self.b2 = b2
        self.batch_size = args.batch_size
        if args.precision==16:
            self.one = torch.tensor(1,dtype=torch.float16).to(self.device)
        else:
            self.one = torch.tensor(1,dtype=torch.float).to(self.device)
        self.mone = -1*self.one
        self.args = args
        # networks
        self.generator_unet = ResNetUNet(args).to(self.device)
        #self.discriminator = Discriminator_CNN(args).to(self.device)
        self.discriminator_unet = MultiDiscriminator(args).to(self.device)
        # for m in self.discriminator_unet.modules():
        #     m.to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.criterion = pytorch_ssim.SSIM().to(self.device)

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
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)
        # sample noise
        #z = torch.randn(imgs.shape[0], self.latent_dim)
        #z = z.type_as(imgs)
        #pass
        lambda_gp = self.args.gp_lambda
        # real_A = Variable(batch["A"],requires_grad=True).to(self.device)
        # real_B = Variable(batch["B"],requires_grad=True).to(self.device)
        # aug_A = Variable(batch["aug_A"],requires_grad=True).to(self.device)
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)
        aug_A = batch["aug_A"].to(self.device)
        if self.args.noise_in:
            noisy = sv.gaussian(real_A,mean=0,stddev=self.args.sigma).to(self.device)
        else:
            noisy = aug_A
        # train generator
        if optimizer_idx == 0:
            frozen_params(self.discriminator_unet)
            free_params(self.generator_unet)

            generated, encoded_, e1_, e2_ = self(real_A)
            noised, z_, z1_, z2_ = self(noisy)
            
            style_loss = (self.args.style_ratio)*(1 - self.criterion(real_B, generated))\
                  + (1-self.args.style_ratio)* self.mse_loss(real_B, generated)
            #enc_loss = args.k_wass *(torch.mean(d_encoded) - torch.mean(d_noise))
            #enc_loss = args.k_wass *(torch.mean(encoded_ - z_)) if args.gram else 0
            enc_loss = self.args.k_wass * (self.mse_loss(encoded_ , z_)+\
                    0.5*self.mse_loss(e1_,z1_) + 0.25*self.mse_loss(e2_,z2_)) if self.args.gram else 0

            
            with autocast():
                h_loss = self.args.k_wass*self.discriminator_unet.forward(generated)
                wass_loss = -torch.mean(h_loss)
           
            self.log("style loss",style_loss)
           
            g_loss = style_loss + enc_loss + wass_loss if self.args.gram else (style_loss + wass_loss)
            self.log("g_loss",g_loss, sync_dist=True)
            self.log("enc loss",enc_loss)
            self.log("wass loss",wass_loss)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            free_params(self.discriminator_unet)
            if self.args.gram or not(self.args.clip_weight):
                free_params(self.generator_unet)
            else:
                frozen_params(self.generator_unet)

            generated, encoded_, e1_, e2_ = self(real_A)
            noised, z_, z1_, z2_ = self(noisy)
            # generated = generated.to(self.device)
            # noised = noised.to(self.device)
            #enc_loss = args.k_wass * (torch.mean(d_encoded) - torch.mean(d_noise))
            #enc_loss = args.k_wass * (torch.mean(encoded_ - z_)) if args.gram else 0
            enc_loss = self.args.k_wass * (self.mse_loss(encoded_ , z_)+\
                    0.5*self.mse_loss(e1_,z1_) + 0.25*self.mse_loss(e2_,z2_)) if args.gram else 0
            self.log("enc loss",enc_loss)        
            with autocast():
                f_loss = self.args.k_wass*self.discriminator_unet.forward(real_B)
                h_loss = self.args.k_wass*self.discriminator_unet.forward(generated)
                d_loss = torch.mean(f_loss) - torch.mean(h_loss) #wasserstein loss
            
            if self.args.clip_weight:
                d_loss -= enc_loss if self.args.gram else 0
                for p in self.discriminator_unet.parameters():
        	        p.data.clamp_(-self.args.clip_value, self.args.clip_value)
            else:
                gradient_penalty = self.compute_gradient_penalty(real_B.data, generated.data)
                d_loss -= enc_loss if self.args.gram else 0
                d_loss += lambda_gp* self.args.k_wass* gradient_penalty
            self.log("discriminator loss",d_loss, sync_dist=True)


            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def backward(self, loss, optimizer, optimizer_idx):
        # do a custom way of backward
        if optimizer_idx == 0:
            #with autocast():
            loss.backward(self.one,retain_graph=True)#.to(self.device)
        else:
            #with autocast():
            loss.backward(self.mone,retain_graph=True)#.to(self.device)


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
        dataset = ImageDataset("../data/%s" % self.args.dataset, input_shape, mode='train')
        return DataLoader(dataset, batch_size=self.args.batch_size,num_workers=7, pin_memory=True)

    def test_dataloader(self):
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        dataset = ImageDataset("../data/%s" % self.args.dataset, input_shape, mode='test')
        return DataLoader(dataset, batch_size= self.args.test_batch_size, shuffle=True, num_workers=1)

    def predict_dataloader(self):
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        dataset = UserDataset("../data/%s" % self.args.dataset, input_shape, mode="user")
        return DataLoader(dataset, batch_size=self.args.test_batch_size, num_workers=1)
        
    def on_epoch_end(self):
        pass

    # def validation_step(self, batch, batch_idx):
    #     pass

class SaveImage(Callback):
    def __init__(self,args) -> None:
        self.args = args

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        input_shape = (self.args.n_channel, self.args.img_height, self.args.img_width)
        dataset = ImageDataset("../data/%s" % self.args.dataset, input_shape, mode='test')
        test_loader = DataLoader(dataset, batch_size= self.args.test_batch_size, shuffle=True, num_workers=1)
        batches_done = pl_module.current_epoch
        pl_module.generator_unet.eval()
        val_loss= sv.sample_images(batches_done, test_loader, self.args, pl_module.generator_unet, pl_module.criterion, Tensor)
        self.log("validation loss",val_loss,sync_dist=True)
        return val_loss

def main(args: Namespace) -> None:
    input_shape = (args.n_channel, args.img_height, args.img_width)
    if args.precision == 16:
        Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
 
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WaeGAN(args)
    dataset = args.dataset
    date = args.date
    save_path = "./save/{dataset}_{date}".format(dataset=dataset,date=date)
    checkpoint_callback = ModelCheckpoint(monitor="g_loss", dirpath=save_path,
    filename="waegan-{epoch:02d}",
    save_top_k=3,
    mode="min",)
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
        ckpt = ModelCheckpoint(dirpath=save_path,filename="waegan-{epoch:02d}")
        base = os.path.basename(ckpt.format_checkpoint_name(dict(epoch=start_epoch)))
        ckpt_path = os.path.join(save_path,base)
        trainer = Trainer(gpus=-1,accelerator=accel,callbacks=callbacks,\
            resume_from_checkpoint=ckpt_path,precision=precision,amp_level='O2',amp_backend="apex",\
                terminate_on_nan = True)
    else:
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
        trainer = Trainer(gpus=-1,accelerator=accel,callbacks=callbacks,\
            precision=precision,amp_level='O2',amp_backend="apex",\
                terminate_on_nan = True)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    if args.train:
        trainer.fit(model)
    else:
        if args.val_target=='train':
            edg.eval_data_gt(args, trainer.forward(model), model.train_dataloader, Tensor, model.mse_loss)
    
        elif args.val_target=='test':
            edg.eval_data_gt(args, trainer.forward(model), model.test_dataloader, Tensor, model.mse_loss)
    
        elif args.val_target=='user':
            ed.eval_data(args, trainer.forward(model), model.predict_dataloader, Tensor)
    
        elif args.val_target=='update':
            ud.update_data(args, trainer.forward(model), Tensor)

        else:
            print("Nothing to be done !")

if __name__ == '__main__':
    
    args = options.Options()
    args = options.Options.parse(args)
    sv.init_imdirs(args)
    main(args)




