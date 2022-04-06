
import argparse
import os

class Options():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        
        parser.add_argument("--n_channel", dest="n_channel", default=3, type=int, help="channels in the input data")
        parser.add_argument("--n_z", dest="n_z", default=32, type=int, help="number of dimensions in latent space")
        parser.add_argument("--sigma", dest="sigma", default=0.25, type=float, help="variance of gaussian noise for encoder training")
        parser.add_argument("--lr", dest="lr", default=0.0002, type=float, help="learning rate for Adam optimizer")
        parser.add_argument("--epochs",dest="epochs", default= 101, type= int, help="how many epochs to run for")
        parser.add_argument("--batch_size", dest="batch_size", default= 32, type= int, help= "batch size for dataset")
        parser.add_argument("--test_batch_size", dest="test_batch_size", default= 16, type= int, help= "batch size for test dataset")
        parser.add_argument("--train_max", dest="train_max", default=300, type= int, help="maximun number of batches for training")
        parser.add_argument("--validate", dest="train", action='store_false', help="flag not to train networks and load networks from saved weights" )
        parser.add_argument("--dataset", dest="dataset", default="SNUH-EGSL-UCUK", type=str, help="name of dataset")
        parser.add_argument("--img_width", dest="img_width", default=384, type=int, help="width of image in pixels")
        parser.add_argument("--img_height", dest="img_height", default=256, type=int, help="height of image in pixels")
        parser.add_argument("--epoch", dest="epoch", default=0, type=int, help="epoch to start")
        parser.add_argument("--date", dest="date", default="111821", type=str, help="id of saved files")
        parser.add_argument("--n_critic", dest="n_critic", default=1, type=int, help="number of skip iteration for generator")
        parser.add_argument("--DDP", dest="DDP", action='store_true', help="flag for distibuted data processing")
        parser.add_argument("--n_resblk", dest="n_resblk", default=1, type=int, help="number of resnet blocks in u-net")
        parser.add_argument("--val_target", dest="val_target", default='test', choices=['train','test','user','update'], type= str, help="target dataset to validation")
        parser.add_argument("--clip_value", dest="clip_value", default=0.01, type=float, help="float number to clip weights")
        parser.add_argument("--no_clip_weight", dest="clip_weight", action='store_false', help="flag not to apply clipping weight")
        parser.add_argument("--gram", dest="gram", action='store_true', help="flag not to count encoder loss")
        parser.add_argument("--gp_lambda", dest="gp_lambda", default= 10, type=float, help="flag not to use swish activation on encoder")
        parser.add_argument("--k_wass", dest="k_wass", default=0.1, type=float, help="A constant to multipy with wasserstein metrics")
        parser.add_argument("--single_critic", dest="multi_critic", action='store_false', help="flag to turn off multiple critic for u-net")
        parser.add_argument("--descending", dest="descending", action='store_true', help="flag to set a resnet network with a shape")
        parser.add_argument("--noise", dest="noise_in", action='store_true', help="flag to disable gaussian noise input")
        parser.add_argument("--disc_channel", dest="disc_channel", default=1, type=int, help="number of channels in rear discriminators")
        parser.add_argument("--dropout", dest="dropout", default= 0.0, type=float, help="float number for dropout probability for U-net downsampling ")
        parser.add_argument("--disc_kernel", dest="disc_kernel", default=3, type=int, help="kernel size of discriminator")
        parser.add_argument("--add_graph", dest="add_graph",action='store_true', help="flag to add tensorboard graph")
        parser.add_argument("--style_ratio", dest="style_ratio", default=0.5, type=float, help ="ratio of mse and style loss")
        parser.add_argument("--nested", dest="nested", action='store_true', help="flag to enable nested u-net")
        parser.add_argument("--no_attention", dest="attention", action='store_false', help="flag to enable attention block for decoder")
        parser.add_argument("--lateral", dest="lateral", action='store_true', help="flag to enable lateral attention layer")
        parser.add_argument("--uncertainty", dest="uncertainty", default=0.05, type=float, help="threshold to ask help")
        parser.add_argument("--precision", dest="precision", default=16, type=int, help="Half or 32bit precision")
        parser.add_argument("--gpu", dest="gpu", default=1, type=int, help="no. of gpus to use")
        parser.add_argument("--n_class", dest="n_class", default=2, type=int, help="class id for validation")
        parser.add_argument("--n_classes", dest="n_classes", default=32, type=int, help="class id for validation")
        parser.add_argument("--last", dest="last", action='store_true', help="flag to use last checkpoint file")
        parser.add_argument("--dataroot", dest="dataroot", default='../../data', type=str, help="root directory of data files")
        parser.add_argument("--ckpt_dir", dest="ckpt_dir", default='./ckpt', type=str, help="root directory of ckpt files")
        parser.add_argument("--ckpt_name", dest="ckpt_name", default='bestmodel.ckpt', type=str, help="name of ckpt files")
        parser.add_argument("--resnet50", dest="resnet50", action='store_true', help="switch to resnet101 encoder")
        parser.add_argument("--fcone", dest="fcone", action='store_true', help="switch to fc layer configuration encoder")
     
        return parser
    
    def gather_options(self):
        
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        #opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt
