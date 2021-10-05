import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
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
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
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
        
        aug_A = self.aug_func(image_A)
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        
        item_augA = self.transform(aug_A)

        return {"A": item_A, "B": item_B, "aug_A": item_augA, "pathA": path_A, "pathB": path_B}

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

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        
        image_A = Image.open(path_A)
        
        # if self.unaligned:
        #     path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        #     image_B = Image.open(path_B)
        # else:
        #     path_B = self.files_B[index % len(self.files_B)]
        #     image_B = Image.open(path_B)
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)
        
        # aug_A = self.aug_func(image_A)
        item_A = self.transform(image_A)
        # item_B = self.transform(image_B)
        
        # item_augA = self.transform(aug_A)

        return {"A": item_A, "pathA": path_A}

    def __len__(self):
        return len(self.files_A) #max, len(self.files_B))
