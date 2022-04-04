import glob
import random
import os
from numpy.lib.function_base import angle

from torch.utils.data import Dataset
from PIL import Image, ImageChops
import torchvision.transforms as transforms
from torchvision import datasets
import post_process as pp
import numpy as np

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode='train', target =2):
        #self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        unaligned=False
        self.unaligned = unaligned
        self.mode = mode
        self.target = target
        self.input_shape = input_shape
        self.files_A = sorted(glob.glob(os.path.join(root, f"{self.mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{self.mode}B") + "/*.*"))
        self.aug_func = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        
        image_A = Image.open(path_A)
        
        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
            image_B = Image.open(path_B)
        else:
            path_B = self.files_B[index % len(self.files_B)]
            image_B = Image.open(path_B)
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        
        if self.mode == 'test':
            aug_A = transforms.functional.affine(img=image_A, angle=15, translate=(0.1, 0.1), scale=(0.9), shear=0.1)
        else:
            aug_A = self.aug_func(image_A)
        #img_tmp = np.asarray(image_B, dtype='uint8')
        #blank_image = np.zeros((self.input_shape[1],self.input_shape[2],self.input_shape[0]), np.uint8)
        #target, _, area_p, _ =pp.critic_segmentation(img_tmp)
        blank = Image.new("RGB", image_B.size, (0,0,0))
        nosignal = Image.new("RGB", image_B.size, (0,0,255))
        dummy = Image.new("RGB", image_B.size, (0,255,255))
        #label = self.target if np.sum(img_tmp) > np.sum(blank_image) else 0
        #diff = ImageChops.difference(image_B, blank)
        if ImageChops.difference(image_B, blank).getbbox():
            if ImageChops.difference(image_B, dummy).getbbox():
                if ImageChops.difference(image_B, nosignal).getbbox():
                    label = self.target
                else:
                    label = 1
            else:
                label = 3
        else:
            label = 0
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        
        item_augA = self.transform(aug_A)

        return {"A": item_A, "B": item_B, "aug_A": item_augA, "pathA": path_A, "pathB": path_B, "label": label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class UserDataset(Dataset):
    def __init__(self, root, input_shape, mode="user"):
        #self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        unaligned=False
        self.unaligned = unaligned
        #pathA = os.path.join(root, "%sA" % mode + "/*.*")
        #pathB = os.path.join(root, "%sB" % mode + "/*.*")
        #list_fileA = glob.glob(pathA)
        #list_fleeB = glob.glob(pathB)
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        # self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        # self.aug_func = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        # self.aug_func = transforms.functional.affine(angle=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        
        image_A = Image.open(path_A)
        
       
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        
        #aug_A = self.aug_func(image_A)
        aug_A = transforms.functional.affine(img=image_A, angle=15, translate=(0.1, 0.1), scale=(0.9), shear=0.1)
        item_A = self.transform(image_A)
        # item_B = self.transform(image_B)
        
        item_augA = self.transform(aug_A)

        return {"A": item_A, "aug_A": item_augA, "pathA": path_A}

    def __len__(self):
        return len(self.files_A) #max, len(self.files_B))
