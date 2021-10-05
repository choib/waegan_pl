import torch
import logging
from torchvision.utils import save_image
from datasets_resnet import *
import sys
import logging
from torch.autograd import Variable
import os

def init_imdirs(args):
    os.makedirs("images/%s_photo_%s" % (args.date ,args.dataset), exist_ok=True)
    os.makedirs("images/%s_seg_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("images/%s_gt_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("images/%s_overlap_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("images/%s_latent_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("images/%s_encoder_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("images/%s_result_%s" % (args.date, args.dataset), exist_ok=True)
    os.makedirs("./save/{}_{}".format(args.dataset, args.date), exist_ok=True)


def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise
  
def sample_images(batches_done, data_loader, args, generator, criterion, Tensor):
    """Saves a generated sample from the test set"""
    imgs = next(iter(data_loader))
    
    real_samples = None
    img_samples = None
    overlap_samples = None
    gt_samples = None
    latent_samples = None
    enc_samples= None
    no_sample = 0
    sum_err = 0
    for img_A, img_B, aug_A in zip(imgs["A"], imgs["B"], imgs["aug_A"]):
        real_A = img_A.view(1, *img_A.shape)
        real_A = Variable(real_A.type(Tensor))
        if args.noise_in:
            noisy = gaussian(real_A.detach(),mean=0,stddev=args.sigma)
        else:
            aug_A = aug_A.view(1, *aug_A.shape)
            aug_A = Variable(aug_A.type(Tensor))
            noisy = aug_A
        #encoded = wae_encoder(real_A)           
        fake_B, latent_img, encoded, _ = generator(real_A)
        real_B = img_B.view(1, *img_B.shape)
        real_B = Variable(real_B.type(Tensor))
        sum_err += 0.5*(1 - criterion(fake_B, real_B))

        overlap = real_A/2 + fake_B/2
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        overlap = torch.cat([x for x in overlap.data.cpu()], -1)
        real_A = torch.cat([x for x in real_A.data.cpu()], -1)
        real_B = torch.cat([x for x in real_B.data.cpu()], -1)
        latent_img = torch.cat([x for x in latent_img.data.cpu()], -1)
        noisy = torch.cat([x for x in noisy.data.cpu()], -1)
        # fake_B = torch.cat(fake_B.data.cpu()[0], -1)
        # overlap = torch.cat(overlap.data.cpu()[0], -1)
        # real_A = torch.cat(real_A.data.cpu()[0], -1)
        # real_B = torch.cat(real_B.data.cpu()[0], -1)
        # latent_img = torch.cat(latent_img.data.cpu()[0], -1)
        # noisy = torch.cat(noisy.data.cpu()[0], -1)
        real_A = torch.cat((real_A,noisy),-1)

        real_samples = real_A if real_samples is None else torch.cat((real_samples, real_A), 1)
        gt_samples = real_B if gt_samples is None else torch.cat((gt_samples, real_B), 1)
        img_samples = fake_B if img_samples is None else torch.cat((img_samples, fake_B), 1)
        overlap_samples = overlap if overlap_samples is None else torch.cat((overlap_samples,overlap),1)
        latent_samples = latent_img if latent_samples is None else torch.cat((latent_samples,latent_img),0)
        enc_samples = encoded if enc_samples is None else torch.cat((enc_samples,encoded),0)
        no_sample += 1
    str = "\tTest: {:<10.6e}".format(sum_err.item()/no_sample)
    sys.stdout.write(str)
    logging.info(str)
    save_image(real_samples, "images/%s_photo_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
    save_image(gt_samples, "images/%s_gt_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
    save_image(img_samples, "images/%s_seg_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
    save_image(overlap_samples, "images/%s_overlap_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
    save_image(latent_samples, "images/%s_latent_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
    save_image(enc_samples, "images/%s_encoder_%s/%s_%s.png" % (args.date, args.dataset, batches_done, no_sample), nrow=1, normalize=True)
   