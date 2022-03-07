from torch.utils.data import DataLoader
import time
import natsort
import pandas as pd
# from models_resnet_pl import *
from models_pl import *
from models_pl_1 import *
from datasets_resnet import *
import options
from pytorch_lightning.core import LightningModule
import os
import cv2
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
low_hsv = [[(0, 100, 100),(170, 100, 100)], #_0
        [(35, 100, 100)], #_1
        [(0, 50, 50)], #_full
        [(0, 50, 50)]] #_uncertaim
high_hsv = [[(10,255,255), (180, 255, 255)], #_0
        [(70, 255, 255)], #_1
        [(180, 255, 255)], #_full
        [(180, 253, 253)]] #_uncertain
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


iter=3
def cnts_extract(im_HSV,low_HSV,high_HSV, erode=False, dilate=False):
    im_threshold = cv2.inRange(im_HSV, low_HSV, high_HSV)
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        im_threshold = cv2.dilate(im_threshold, kernel, anchor=(0, 0), iterations=iter)
    if erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        im_threshold = cv2.erode(im_threshold, kernel, anchor=(0, 0), iterations=iter)
    gray = cv2.bitwise_and(im_HSV,im_HSV,mask=im_threshold)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, reverse=True, key=lambda x: cv2.contourArea(x))
    area = 0
    if len(contours) > 0:
        for cnt in contours:
            area += cv2.contourArea(cnt)
    else:
        contours = None
        #area = 0
    return contours, area


def critic_segmentation_by_class(id, im_seg,):

    im_seg = cv2.cvtColor(im_seg, cv2.COLOR_RGB2BGR)
    seg_HSV = cv2.cvtColor(im_seg, cv2.COLOR_BGR2HSV)
    seg_area = 0.0
    seg_c = []
    uncertain_area = 0.0
    for j in range(len(low_hsv[id])):
        seg_cnts, s_area = cnts_extract(seg_HSV, low_hsv[id][j], high_hsv[id][j], dilate=True, erode=True)
        seg_area += s_area
        seg_c.append(seg_cnts)
        # u_h_hsv = high_hsv[id][j]
        # (h, s, v) = high_hsv[id][j]
        # u_h_hsv = (h, 200, 200)
        # print(low_hsv[-1][j], high_hsv[-1][j])
        u_seg_cnts, u_seg_area = cnts_extract(seg_HSV, low_hsv[-1][j], high_hsv[-1][j], dilate=True, erode=True)
        uncertain_area += u_seg_area
    return seg_area,uncertain_area




#might need to edit this funstion to give the desired frames...
def video_to_image(src,dst,time_skip_rate):
    cap = cv2.VideoCapture(src)
    frame_list=[]
    fps = cap.get(cv2.CAP_PROP_FPS)

    clean_dir(dst)
    # Get the total numer of frames in the video.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
    success, image = cap.read()
    print("Converting video to image......")
    while success and frame_number <= frame_count:

        # do stuff
        frame_list.append(frame_number)
        image = cv2.resize(image, (384, 256))
        cv2.imwrite(dst+"/"+str(frame_number) + ".png", image)

        frame_number += int(fps*time_skip_rate)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = cap.read()
    print("Conversion complete........")
    return frame_list
class ImageDataset(Dataset):
    def __init__(self, root, input_shape):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        unaligned = False
        self.unaligned = unaligned
        self.files_A = natsort.natsorted(glob.glob(os.path.join(root + "/*.*")),reverse=False)
        self.aug_func = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)


    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        image_A = Image.open(path_A)

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        item_A = self.transform(image_A)
        aug_A = transforms.functional.affine(img=image_A, angle=15, translate=(0.1, 0.1), scale=(0.9), shear=0.1)
        item_augA = self.transform(aug_A)
        return {"A": item_A,"aug_A":item_augA}

    def __len__(self):
        return len(self.files_A)


# class WaeGAN(LightningModule):
#
#     def __init__(self, args):
#         super().__init__()
#         self.save_hyperparameters()
#         self.latent_dim = args.n_z
#         self.lr = args.lr
#         self.n_critic = args.n_critic
#         self.args = args
#         self.smth = 0.45  # args.smth
#         # self.automatic_optimization=False
#         # self.b1 = b1
#         # self.b2 = b2
#         self.batch_size = args.batch_size
#         self.one = torch.tensor(1, dtype=torch.float)  # .to(self.device)
#         # if args.precision==16:
#         #     self.one = self.one.half()
#         # else:
#         #     pass
#         # self.mone = -1*self.one
#
#         # self.df_csv = f"./csv/{args.date}_{args.dataset}.csv"
#         # self.tmp_csv = f"./tmp/{args.date}_{args.dataset}_result.csv"
#         # self.tmp_pred = f"./tmp/{args.date}_{args.dataset}_predict.csv"
#         # networks
#         self.generator_unet = ResNetUNet(args)  # .to(self.device)
#         self.discriminator_unet = MultiDiscriminator(args)  # .to(self.device)
#         self.mse_loss = nn.MSELoss()  # .to(self.device)
#         self.adv_loss = torch.nn.BCEWithLogitsLoss()  # .to(self.device)
#         # self.aux_loss = LabelSmoothingCrossEntropy()  # LabelSmoothing(self.smth)#torch.nn.CrossEntropyLoss(label_smoothing=self.smth)#
#         # self.criterion = pytorch_ssim.SSIM()  # .to(self.device)
#         self.no_sample = 0
#         self.sum_test = 0
#         # self.json_dir = f"./tmp/{args.date}_{args.dataset}_json"
#         # self.jpg_dir = f"./tmp/{args.date}_{args.dataset}_jpg"
#         # self.png_dir = f"./tmp/{args.date}_{args.dataset}_png"
#         # os.makedirs(self.json_dir, exist_ok=True)
#         # os.makedirs(self.jpg_dir, exist_ok=True)
#         # os.makedirs(self.png_dir, exist_ok=True)
#         # clean_dir(self.json_dir)
#         # clean_dir(self.jpg_dir)
#         # self.result = []
#         self.inv_transform = transforms.Compose(
#             [
#                 UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#                 transforms.ToPILImage(),
#             ]
#         )
#
#     def forward(self, z):
#         return self.generator_unet(z)

class WaeGAN(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = args.n_z
        self.lr = args.lr
        self.n_critic = args.n_critic
        self.batch_size = args.batch_size
        self.one = torch.tensor(1, dtype=torch.float)  # .to(self.device)
        self.args = args
        # networks
        self.generator_unet = ResNetUNet(args)  # .to(self.device)
        self.discriminator_unet = MultiDiscriminator(args)  # .to(self.device)

    def forward(self, z):
        return self.generator_unet(z)
class WaeGAN123(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = args.n_z
        self.lr = args.lr
        self.n_critic = args.n_critic
        self.batch_size = args.batch_size
        self.one = torch.tensor(1, dtype=torch.float)  # .to(self.device)
        self.args = args
        # networks
        self.generator_unet = ResNetUNet1(args)  # .to(self.device)
        self.discriminator_unet = MultiDiscriminator1(args)  # .to(self.device)

    def forward(self, z):
        return self.generator_unet(z)
# class WaeGAN1(LightningModule):
#
#     def __init__(self, args):
#         super().__init__()
#         self.save_hyperparameters()
#         self.latent_dim = args.n_z
#         self.lr = args.lr
#         self.n_critic = args.n_critic
#         self.batch_size = args.batch_size
#         self.one = torch.tensor(1, dtype=torch.float)  # .to(self.device)
#         self.args = args
#         # networks
#         self.generator_unet = ResNetUNet(args)  # .to(self.device)
#         self.discriminator_unet = MultiDiscriminator(args)  # .to(self.device)
#
#     def forward(self, z):
#         return self.generator_unet(z)



data_dict={
            'SNUH-FNRL-UCCV':[-0.227,0.003],
            'SNUH-GAUL-UCUK':[0.093,0.005],
            'SNUH-GRSL-DBUK':[0.495,0.008],
            'SNUH-GRSL-STNT':[-0.084,0.012],
            'SNUH-HMSL-UCET':[-0.463,0.005],
            'SNUH-JPCL-UCUK':[0.114,0.008],
            'SNUH-SCSL-UCAE':[0.123,0.020],
            'SNUH-SCSL-UCET':[-0.510,0.005],
            'SNUH-SIRL-UCET':[-0.134,0.005]}
if __name__ == '__main__':
    start_time=time.time()
    args = options.Options()
    args = options.Options.parse(args)
    inv_transform = transforms.Compose(
        [
            UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage(),
        ]
    )

    videos=os.listdir("videos")
    for video in videos:
        src="/home/mteg-vas/nisan/workspace/wgan_git/data/videos/"
        dst="temporary"
        time_skip_rate=0.5 #time to skip while extracting frames from videos in seconds

        frame_list=video_to_image("videos/"+video,dst,time_skip_rate)


        input_shape = (args.n_channel, args.img_height, args.img_width)



        # model_list=os.listdir("19_models_inference_mix")
        model_list=os.listdir("/home/mteg-vas/nisan/workspace/wgan_git/WGAN/SNUH-mix")
        # model_list=os.listdir("25_models_professor")
        # print(model_list)
        model_list.sort()
        print(model_list)
        dataset = ImageDataset("temporary", input_shape)
        # dataset = ImageDataset("/home/mteg-vas/nisan/workspace/wgan_git/data/temporary/newA", input_shape)
        # dataset = ImageDataset("/home/mteg-vas/nisan/workspace/wgan_git/data/ICT_10_26_mix_dataset/Polymer Clip Applier (CLAL-UCUK)/testA", input_shape)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)# the images are sorted in the dataset
        loss=nn.MSELoss()
        results_dict = {'Frame_No.':frame_list}
        # results_dict = {}
        for equipment_name in model_list:
            # checkpoint path of the model
            # ckpt_path="/home/mteg-vas/nisan/workspace/wgan_git/WGAN/19_models_inference_mix/"+equipment_name+"/weight.ckpt"
            ckpt_path="/home/mteg-vas/nisan/workspace/wgan_git/WGAN/SNUH-mix/"+equipment_name+"/weight.ckpt"
            # ckpt_path="/home/mteg-vas/nisan/workspace/wgan_git/WGAN/25_models_professor/"+equipment_name

            model = WaeGAN(args)



            model = model.load_from_checkpoint(ckpt_path,strict=False)
            model.cuda()
            model.eval()
            detected_list=[] #1 if detected and 0 if not detected

            start_time_equipment = time.time()
            count=0
            area_p_list=[]
            data = data_dict.get(equipment_name)
            for i, batch in enumerate(test_loader):
                # start_time_equipment = time.time()

                real_A = Variable(batch["A"]).type(Tensor)  # .cuda()
                # aug_A = Variable(batch["aug_A"]).type(Tensor)  # .cuda()
                fake_B, e, e1, e2 = model(real_A)
                # _, z, z1, z2 = model(aug_A)
                fake_B = fake_B.data[0]
                # print(e2)
                # nz_f = loss(e, z)  # + self.mse_loss(e1, z1) + self.mse_loss(e2, z2)
                # nz_f = nz_f.item()
                e2=e2.item()

                img_seg = inv_transform(fake_B)
                img_seg = np.asarray(img_seg, dtype='uint8')
                # seg_area,uncertain_area=critic_segmentation_by_class(1,img_seg)
                target, area, area_p, cnts = pp.critic_segmentation(img_seg)

                if e2>= (data[0] - data[1]) and e2<=data[0]+data[1]:
                    detected_list.append(1)
                else:
                    detected_list.append(0)
                # if area_p >0.4 :
                #     detected_list.append(1)
                # else:
                #     detected_list.append(0)
                # area_p_list.append(area_p)

            # print(equipment_name, "  average = ", sum(area_p_list)/len(area_p_list))
            print(equipment_name, "  detected = ", sum(detected_list)," Total = ",len(test_loader))
            print("--- %s seconds ---" % (time.time() - start_time_equipment))
            results_dict[equipment_name]=detected_list
            # print(detected_list)
        dataframe=pd.DataFrame.from_dict(results_dict)
        dataframe.to_csv("results/"+video.split(".")[0]+".csv",index=False,header=True)
        print("--- %s seconds ---" % (time.time() - start_time))