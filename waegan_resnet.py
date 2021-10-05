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

from models_resnet_m import *
from datasets_resnet import *

import torch.distributed as dist
from parallel import DataParallelModel as DPM 
from parallel import DataParallelCriterion as DPC

import pytorch_ssim
import options
import save_im as sv
import print_sysout as ps
import eval_data_gt as edg
import eval_data as ed
import update_data as ud

args = options.Options()
args = options.Options.parse(args)

n_skip_iter = args.n_critic
input_shape = (args.n_channel, args.img_height, args.img_width)
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(filename="./%s.log" % args.date ,format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info(args)
logging.getLogger('PIL').setLevel(logging.WARNING)
# for DDP
if args.DDP:
    
    args.gpu = 0
    args.local_rank = 3
    args.world_size = 1
    
    #args.gpu = args.local_rank
    #torch.cuda.set_device(args.gpu)
    #torch.distributed.init_process_group(backend='nccl',
    #                                     init_method=None)
    #args.world_size = torch.distributed.get_world_size()

sv.init_imdirs(args)
train_loader = DataLoader(
    ImageDataset("../../data/%s" % args.dataset, input_shape),
    batch_size=args.batch_size if args.train else 1,
    shuffle=True,
    num_workers=8,
)

test_loader = DataLoader(
    ImageDataset("../../data/%s" % args.dataset, input_shape, mode="test"),
    batch_size=args.test_batch_size if args.train else 1,
    shuffle=True,
    num_workers=1,
)        

if args.val_target == 'user':
    user_loader = DataLoader(
        UserDataset("../../data/%s" % args.dataset, input_shape, mode="user"),
        batch_size=args.test_batch_size if args.train else 1,
        shuffle=True,
        num_workers=1,
    )       
# instantiate discriminator model, and restart encoder and decoder, for fairness. Set to train mode, etc
generator_unet, discriminator, discriminator_unet = ResNetUNet(args), Discriminator_CNN(args), MultiDiscriminator(args)
mse_loss = nn.MSELoss()
criterion = pytorch_ssim.SSIM()
#criterion = pytorch_msssim.MSSSIM()
writer = SummaryWriter("./save/{}_{}/WAEgan_Summary".format(args.dataset, args.date))


if cuda:
    generator_unet.cuda() 
    discriminator.cuda()
    discriminator_unet.cuda()
    criterion.cuda()
   
if args.DDP:
    
    generator_unet.cuda(args.gpu) 
    discriminator.cuda(args.gpu)
    discriminator_unet.cuda(args.gpu)
    criterion.cuda(args.gpu)
    
    generator_unet = DPM(generator_unet) 
    discriminator = DPM(discriminator)   
    discriminator_unet = DPM(discriminator_unet)
    criterion = DPC(criterion)
   
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if args.epoch !=0:
    # Load pretrained models
    start_epoch = args.epoch - 1
    generator_unet.load_state_dict(torch.load("./save/{}_{}/WAEgan_{}-epoch_{}.pth".format(args.dataset, args.date,'decoder', start_epoch)))
    if args.train:
        if args.disc_retrain:
            discriminator.apply(weights_init_normal)
            discriminator_unet.apply(weights_init_normal)
        else:
            discriminator.load_state_dict(torch.load("./save/{}_{}/WAEgan_{}-epoch_{}.pth".format(args.dataset, args.date,'discriminator', start_epoch)))
            discriminator_unet.load_state_dict(torch.load("./save/{}_{}/WAEgan_{}-epoch_{}.pth".format(args.dataset, args.date,'discriminator_C', start_epoch)))
        
else:
    #wae_encoder.apply(weights_init_normal)
    generator_unet.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    discriminator_unet.apply(weights_init_normal)

if args.train:
    gen_optim = torch.optim.Adam(generator_unet.parameters(), lr = args.lr)
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr = args.lr)
    dis_u_optim = torch.optim.Adam(discriminator_unet.parameters(), lr =args.lr)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optim, step_size=30, gamma=0.5)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)
    dis_u_scheduler = torch.optim.lr_scheduler.StepLR(dis_u_optim, step_size=30, gamma=0.5)

    # one and -one allow us to control descending / ascending gradient descent
    one = torch.tensor(1, dtype=torch.float)
    one = one.cuda()
    mone = one * -1
    
    prev_time = time.time()
    for epoch in range(args.epoch,args.epochs):

        # train for one epoch -- set nets to train mode
        #wae_encoder.train()
        generator_unet.train()
        discriminator.train()
        discriminator_unet.train()
        
        # Included are elements similar to the Schelotto (2018) implementation
        # on GitHub. Schelotto's implementation repository is worth looking into, 
        # because the WAE-MMD ("Maximum Mean Discrepancy") implementation, a second 
        # WAE algorithm discussed in the original Wasserstein Auto-Encoders paper,
        # is also implemented there.
        
        for i, batch in enumerate(train_loader):
            # zero gradients for each batch
            
            generator_unet.zero_grad()
            discriminator.zero_grad()
            discriminator_unet.zero_grad()
            if i > args.train_max:
                break

           
            #### TRAIN DISCRIMINATOR ####

            # flip which networks are frozen, which are not
            free_params(discriminator_unet)
            frozen_params(generator_unet)
            if args.ex_critic:
                free_params(discriminator)
                
            else:
                frozen_params(discriminator)
                

            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            aug_A = Variable(batch["aug_A"].type(Tensor))
            overlap = real_A/2 + real_B/2
            # make noisy images
            if args.noise_in:
                noisy = sv.gaussian(real_A.detach(),mean=0,stddev=args.sigma)
            else:
                noisy = aug_A
                #noisy = overlap
            
            
            generated, encoded_, encoded, d_encoded = generator_unet(real_A.detach())
            noised, z_, z, d_noise = generator_unet(noisy.detach())
            #
            if args.add_graph:
                writer.add_graph(generator_unet, real_A)
            
            # run discrimators for real and fake images
            if args.ex_critic:
                d_encoded = discriminator(encoded_)
                d_noise = discriminator(z_)
            
                #writer.add_graph(discriminator, encoded)
            # calculate discriminator loss
            if args.wass_metric:
                if args.ex_critic:
                    #wass_enc_loss = args.k_wass * (d_encoded - d_noise).mean()
                    wass_enc_loss = args.k_wass * (d_encoded.mean() - d_noise.mean())
                    e_loss = args.k_wass*d_noise.mean()
                    if args.gan:
                        wass_enc_loss.backward(mone,retain_graph=True)
                else:
                    wass_enc_loss = args.k_wass * (encoded - z).mean()
                    e_loss = args.k_wass*z.mean()
                     
            else:    
                d_loss = (torch.log((d_encoded)).mean())
            # calculate gradient and propagte to MAXIMIZE discriminator output
                e_loss = (torch.log((1-d_noise)).mean())
                #for printing status
                wass_enc_loss = args.k_wass * (d_loss - e_loss)
            # to MAXIMIZE discriminator loss
                if args.gan:
                    e_loss.backward(mone,retain_graph=True)
                    d_loss.backward(mone,retain_graph=True)
           
            if args.multi_critic and args.gan:
                #writer.add_graph(discriminator_unet, real_B)
                f_loss = args.k_wass*discriminator_unet.compute_out(real_B)
                h_loss = args.k_wass*discriminator_unet.compute_out(generated)
                wass_loss = (f_loss - h_loss).mean()
                h_loss = h_loss.mean()
                wass_loss.backward(mone,retain_graph=True)
            else:
                f_loss = h_loss = wass_loss = None
            
            if not args.clip_weight and args.gan:
                gp_loss = args.n_channel * args.gp_lambda *args.k_wass* discriminator_unet.compute_gradient_penalty(real_B,generated) 
                #if args.ex_critic and args.multi_critic:
                #    gp_loss += args.gp_lambda *args.k_wass* discriminator.compute_gradient_penalty(encoded_,z_)
                #else:
                #    gp_loss = args.gp_lambda *args.k_wass* discriminator.compute_gradient_penalty(encoded,z)   
                gp_loss = gp_loss.mean()
                gp_loss.backward(one)
            else:
                gp_loss = None
            # update discriminator
            #
            if args.gan and args.ex_critic:
                dis_optim.step()
            if args.multi_critic and args.gan:
                dis_u_optim.step()
            # Weight Clipping
            if args.clip_weight and args.gan:
                #for p in generator_unet.parameters():
                #    p.data.clamp_(-args.clip_value, args.clip_value)
                for p in discriminator_unet.parameters():
        	        p.data.clamp_(-args.clip_value, args.clip_value)
                if args.ex_critic:    
                    for p in discriminator.parameters():
        	            p.data.clamp_(-5.0*args.clip_value, 5.0*args.clip_value)
                
        	        
            #### TRAIN Generator/Decoder ####
            #generator_unet.zero_grad()
            if (i % args.n_critic == 0):
                # free auto encoder params
                frozen_params(discriminator_unet)
                frozen_params(discriminator)
                free_params(generator_unet)
                #noisy = overlap # or overlap for endoscope 

                noised, z_, z, d_noise = generator_unet(noisy)
                generated, encoded_, encoded, d_encoded = generator_unet(real_A)
                if args.ex_critic:    
                    d_encoded = discriminator(encoded_)
                    d_noise = discriminator(z_)
                  
                if args.mse:
                    g_loss = mse_loss(real_B, generated) 
                else:
                    g_loss = (args.style_ratio)*(1 - criterion(real_B, generated)) + (1-args.style_ratio)*mse_loss(real_B, generated)
                    #g_loss = (args.style_ratio)*(mse_loss(overlap, generated)) + (1-args.style_ratio)*mse_loss(real_B, generated)
                style_loss =  g_loss #real_B
                # calaulate propagate gradients to MINIMIZE generator loss
                if args.wass_metric:
                    if args.ex_critic:
                        #wass_enc_loss = args.k_wass *(encoded - z).mean()
                        #wass_enc_loss += args.k_wass *(d_encoded - d_noise).mean()#(encoded - z).mean()
                        enc_loss = args.k_wass *(encoded.mean() - z.mean())
                        wass_enc_loss = args.k_wass *(d_encoded.mean() - d_noise.mean())
                        e_loss =  args.k_wass*d_noise.mean()
                    else:
                        wass_enc_loss = args.k_wass *(encoded - z).mean()#(encoded - z).mean()
                        e_loss =  args.k_wass*z.mean()
                
                else:
                    e_loss = args.k_wass*(torch.log((d_noise)).mean())
                #if args.gan:     
                wass_enc_loss.backward(one,retain_graph=True)
                # if args.enc_loss: 
                #     enc_loss.backward(one,retain_graph=True) 
                
            #compute generator loss and minimize it
                if args.mse:
                    g_loss.backward(one,retain_graph=True)  
                else:
                    style_loss.backward(one, retain_graph=True)    
         
                if args.multi_critic and args.gan:
                    f_loss = args.k_wass * discriminator_unet.compute_out(real_B)#real_B 
                    h_loss = args.k_wass * discriminator_unet.compute_out(generated)
                    wass_loss = f_loss.mean() - h_loss.mean()
                    h_loss = h_loss.mean()
                    wass_loss.backward(one,retain_graph=True)             
                

            # update parameters for generator and encoders 
                gen_optim.step()
                
            lloader = len(train_loader)
            batches_left = args.epochs * lloader - (epoch * lloader + i)
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            writer.add_scalar('training_loss', g_loss, epoch)
            str = ps.pr_status(args, epoch, lloader, i, g_loss, e_loss, wass_enc_loss, wass_loss, h_loss, time_left)
        logging.info(str)     
        #wae_encoder.eval()
        generator_unet.eval()
        real_samples = None
        img_samples = None
        overlap_samples = None
        gt_samples = None
        batches_done = epoch
        
        #for images, _ in tqdm(test_loader):
        #
        
        if args.save:
            if (epoch % args.cpt_interval == 0):
                save_path = './save/{}_{}/WAEgan_{}-epoch_{}.pth'
                #torch.save(wae_encoder.state_dict(), save_path.format(args.dataset, args.date,'encoder', epoch))
                torch.save(generator_unet.state_dict(), save_path.format(args.dataset, args.date,'decoder', epoch))
                torch.save(discriminator.state_dict(), save_path.format(args.dataset, args.date,'discriminator', epoch))
                torch.save(discriminator_unet.state_dict(), save_path.format(args.dataset, args.date,'discriminator_C', epoch))
            if len(test_loader) >0:
                sv.sample_images(batches_done, test_loader, args, generator_unet, criterion, Tensor)
        
else:
    if args.val_target=='train':
        edg.eval_data_gt(args, generator_unet, train_loader, Tensor, mse_loss)
    
    elif args.val_target=='test':
        edg.eval_data_gt(args, generator_unet, test_loader, Tensor, mse_loss)
    
    elif args.val_target=='user':
        ed.eval_data(args, generator_unet, user_loader, Tensor)
    
    elif args.val_target=='update':
        ud.update_data(args, generator_unet, Tensor)

    else:
        print("Nothing to be done !")
        