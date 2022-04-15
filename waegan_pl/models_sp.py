

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101




def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Batch") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# from fast_neural_style    
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.25):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # target = target.float() * (self.confidence) + 0.5 * self.smoothing
        # return F.mse_loss(x, target.type_as(x))
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)

#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

#     G = torch.mm(features, features.t())  # compute the gram product

#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)

# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )
#https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int, lat=False):
        super(AttentionBlock,self).__init__()
        self.lateral = lat
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
            )
        
        if self.lateral:
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
            )
        else:
            self.W_x = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
            )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
            
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(F_l, F_g,  kernel_size=1,stride=1,padding=0,bias=True)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x0):
        if self.lateral:
            x = x0
        else:    
            x = self.up(x0)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ResidualBlock(nn.Module):
    def __init__(self, in_features, relu_act=True, batch_norm=False, swish_act=False):
        super(ResidualBlock, self).__init__()
        layers = [nn.ReflectionPad2d(1)]
        layers.append(nn.Conv2d(in_features, in_features, 3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_features, 0.8))
        else:
            layers.append(nn.InstanceNorm2d(in_features, affine=True))
        if relu_act:
            if swish_act:
                layers.append(Swish())
            else:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(in_features, in_features, 3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_features, 0.8))
        else:
            layers.append(nn.InstanceNorm2d(in_features, affine=True))
        
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.block(x)

class nResNet(nn.Module):
    def __init__(self, num_residual_blocks, out_features, relu_act=True, batch_norm=False, swish_act=False):
        super(nResNet, self).__init__()

        model = [
        ]
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features, relu_act, batch_norm, swish_act)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, swish_act=False, relu_act=True, style=False):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if style:
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8)) 
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:
                    layers.append(nn.LeakyReLU(0.2,inplace=True))
         
            layers.append(nn.Conv2d(out_size, out_size, 3, stride=1, padding=1))
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8)) 
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))     
        else:
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))   
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:
                    layers.append(nn.LeakyReLU(0.2,inplace=True))    
        #else:
        #    
        # if relu_act:
        #     if swish_act:
        #         layers.append(Swish())
        #     else:
        #         layers.append(nn.LeakyReLU(0.2))
         
        layers.append(nn.Dropout2d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, relu_act=True, swish_act=False, full_style=False):
        super(UNetUp, self).__init__()
        layers = [nn.Upsample(scale_factor=2)]
        layers.append(nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if full_style:
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:
                    layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_size, out_size, 3, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if relu_act:
            if swish_act:
                layers.append(Swish())
            else:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Swish(nn.Module):
    """Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.Swish()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        p = torch.sigmoid(input)
        p = p.mul(input)
        return p

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


def swish(x):
    return x * torch.sigmoid(x)

class Mish(nn.Module):

    def forward(self, input):
        p = torch.tanh(F.softplus(input))
        p = p.mul(input)
        return p

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


def mish(x):
    return x * torch.tanh(F.softplus(x))


class ResNetUNetEncoder(nn.Module):
    def __init__(self, args):
        super(ResNetUNetEncoder, self).__init__()

        self.n_channel = args.n_channel
        self.img_height = args.img_height
        self.img_width = args.img_width
       
        self.descending = args.descending
        self.resnet50 = args.resnet50
        self.dropout = args.dropout
        self.batch_size = args.batch_size
        
        
        self.attention = args.attention
        self.nested = args.nested
        self.lateral = args.lateral
        self.fcone = args.fcone
       
        #
        if self.resnet50:
            #self.base_model = resnet50(pretrained=False)
            self.base_model = resnet101(pretrained=False)
        else:
            self.base_model = resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())  

        #latent_dim = self.n_z
        channels = self.n_channel
        
        self.n_classes = args.n_classes
        self.print = Print()
       
        self.layer1 = nn.Sequential(*self.base_layers[:3]) # 
        self.layer2 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer3 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)  
        self.layer4 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16) 
        self.layer5 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        #self.layer6 = nn.Sequential(nn.Conv2d(512, 1, 3, stride=2, padding=1), Swish(), nn.AdaptiveAvgPool2d(1))
        #else:
        self.down1 = UNetDown(channels, 64, normalize=False)  
        self.down2 = UNetDown(64 + 64, 128)
       
        if self.resnet50:
            self.down3 = UNetDown(128 + 256, 256)
            self.down4 = UNetDown(256 + 512, 512)
            self.down5 = UNetDown(512 + 1024, 512)
            self.down6 = UNetDown(512 + 2048, 512)
        else:
            self.down3 = UNetDown(128 + 64, 256)
            self.down4 = UNetDown(256 + 128, 512)
            self.down5 = UNetDown(512 + 256, 512)
            self.down6 = UNetDown(512 + 512, 512)
            
        self.down7 = UNetDown(512, 512)
        
        if self.resnet50:
            self.pooling = nn.Sequential(nn.Flatten(), nn.Dropout(0.0),nn.Linear(2048*8*12, 512), nn.LeakyReLU(0.2,inplace=True))#2048 for resnet101 512 for else
        else:
            self.pooling = nn.Sequential(nn.Flatten(), nn.Dropout(0.0),nn.Linear(512*8*12, 512), nn.LeakyReLU(0.2,inplace=True))#2048 for resnet101 512 for else
        self.fc = nn.Sequential(nn.Linear(512, self.n_classes))#, nn.LeakyReLU(0.2,inplace=True)) # resnet18: 256
        self.critic = nn.Linear(self.n_classes, 1) 
        self.fc1 = nn.Linear(512, 1) 
        
    def forward(self, x):
        x = x

        #d0 = self.down0(x)
        l1 = self.layer1(x)
        d1 = self.down1(x)
        
        mz = torch.cat((l1,d1),1)
        #if self.img_add:
        #d1 = (d1 + l1)/2
        l2 = self.layer2(l1)
        d2 = self.down2(mz)

        
        mz = torch.cat((l2,d2),1)
        l3 = self.layer3(l2)
        d3 = self.down3(mz)

        mz = torch.cat((l3,d3),1)
        l4 = self.layer4(l3)
        d4 = self.down4(mz)

        mz = torch.cat((l4,d4),1)
        l5 = self.layer5(l4)
        d5 = self.down5(mz)

        # self.print(l5)
        l6 = self.pooling(l5)
        # self.print(l6)
        l7 = self.fc(l6)
      
        if self.fcone:
            l8 = self.fc1(l6)
        else:
            l8 = self.critic(l7)
        mz = torch.cat((l5,d5),1)
        d6 = self.down6(mz)
        
        #d7 = self.down7(d6)

        downstream = [d1,d2,d3,d4,d5,d6,l7,l8]
       
        return downstream

class ResNetUNetDecoder(nn.Module):
    def __init__(self, args):
        super(ResNetUNetDecoder, self).__init__()

        self.n_channel = args.n_channel
        #self.n_z = args.n_z
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.descending = args.descending
        self.dropout = args.dropout
        #self.resnet50 = args.resnet50
        self.batch_size = args.batch_size
        self.attention = args.attention
        self.nested = args.nested
        self.lateral = args.lateral
        self.n_z = args.n_z
        channels = self.n_channel
        no_resblk = args.n_resblk
        self.n_classes = args.n_classes
        self.print = Print()
       
        self.down7 = UNetDown(512, 512)
        if self.nested: 
            self.up1 = UNetUp(512, 512)
            self.res1 = nResNet(no_resblk, 512)
            self.up2 = UNetUp(512+512, 512)
            self.res2 = nResNet(no_resblk, 512)
            self.res2_1 = nResNet(no_resblk, 512+512)
            self.up3 = UNetUp(1024+512, 256)
            self.up3_1 = UNetUp(512, 256)
            self.up3_2 = UNetUp(1024, 256)   
            self.res3 = nResNet(no_resblk, 512)
            self.res3_1 = nResNet(no_resblk, 512+256)
            self.res3_2 = nResNet(no_resblk, 512+512)
            self.up4 = UNetUp(1024 + 256, 128)
            self.up4_1 = UNetUp(512, 128)
            self.up4_2 = UNetUp(512 + 256, 128)
            self.up4_3 = UNetUp(512 + 512, 128)         
            self.res4 = nResNet(no_resblk, 256)
            self.res4_1 = nResNet(no_resblk, 256+128)
            self.res4_2 = nResNet(no_resblk, 256+256)
            self.res4_3 = nResNet(no_resblk, 256+384)
            self.up5 = UNetUp(256 + 512, 64)
            self.up5_1 = UNetUp(256, 64)
            self.up5_2 = UNetUp(256 + 128, 64)
            self.up5_3 = UNetUp(256 + 256, 64)
            self.up5_4 = UNetUp(256 + 256 + 128, 64)
            self.res5 = nResNet(no_resblk, 128) 
            self.res5_1 = nResNet(no_resblk, 128+64) 
            self.res5_2 = nResNet(no_resblk, 128+128) 
            self.res5_3 = nResNet(no_resblk, 128+192) 
            self.res5_4 = nResNet(no_resblk, 128+256) 
            self.up6 = UNetUp(448, 64)
            self.up6_1 = UNetUp(128, 64)
            self.up6_2 = UNetUp(128 + 64, 64)
            self.up6_3 = UNetUp(128 + 128, 64)
            self.up6_4 = UNetUp(128 + 128 + 64, 64)
            self.up6_5 = UNetUp(128 + 128 + 64 + 64, 64)
            self.res6 = nResNet(no_resblk, 64) 
            self.res6_1 = nResNet(no_resblk, 64+64) 
            self.res6_2 = nResNet(no_resblk, 64+128) 
            self.res6_3 = nResNet(no_resblk, 64+192) 
            self.res6_4 = nResNet(no_resblk, 64+256) 
            self.res6_5 = nResNet(no_resblk, 64+320) 
       
        elif self.attention:
            if self.lateral==True:
                self.up1 = UNetUp(512, 512)
                self.res1 = nResNet(no_resblk * 8, 512)
                self.att1 = AttentionBlock(1024,512,512,lat=True)
                self.up2 = UNetUp(1024, 512)
                self.res2 = nResNet(no_resblk * 8, 512)
                self.att2 = AttentionBlock(1024, 512, 512,lat=True)
                self.up3 = UNetUp(1024, 256)
                self.res3 = nResNet(no_resblk * 8, 512)
                self.att3 = AttentionBlock(512+256, 512, 512, lat=True)
                self.up4 = UNetUp(768, 128)
                self.res4 = nResNet(no_resblk * 4, 256)
                self.att4 = AttentionBlock(256+128,256,256, lat=True)
                self.up5 = UNetUp(384, 64)
                self.res5 = nResNet(no_resblk * 2, 128) 
                self.att5 = AttentionBlock(128+64,128,128, lat=True)
                self.up6 = UNetUp(192, 64)
                self.res6 = nResNet(no_resblk * 1, 64) 
                self.att6 = AttentionBlock(64+64,64,64, lat=True)
            else:    
                self.up1 = UNetUp(512, 512)
                self.res1 = nResNet(no_resblk * 8, 512)
                self.att1 = AttentionBlock(512,512,512)
                self.up2 = UNetUp(1024, 512)
                self.res2 = nResNet(no_resblk * 8, 512)
                #self.res21 = nResNet(no_resblk * 8, 512)
                self.att2 = AttentionBlock(512, 1024 ,512)
                self.up3 = UNetUp(1024, 256)
                self.res3 = nResNet(no_resblk * 8, 512)
                #self.res31 = nResNet(no_resblk * 8, 512)
                self.att3 = AttentionBlock(512, 1024, 512)
                self.up4 = UNetUp(768, 128)                
                self.res4 = nResNet(no_resblk * 4, 256)
                #self.res41 = nResNet(no_resblk * 4, 256)
                self.att4 = AttentionBlock(256,768,256)
                self.up5 = UNetUp(384, 64)
                self.res5 = nResNet(no_resblk * 2, 128) 
                #self.res51 = nResNet(no_resblk * 2, 128) 
                self.att5 = AttentionBlock(128,384,128)
                self.up6 = UNetUp(192, 64)
                self.res6 = nResNet(no_resblk * 1, 64) 
                #self.res61 = nResNet(no_resblk * 1, 64) 
                self.att6 = AttentionBlock(64,192,64)
        else: 
            self.up1 = UNetUp(512, 512)
            self.res1 = nResNet(no_resblk * 8, 512)
            self.up2 = UNetUp(1024, 512)
            self.res2 = nResNet(no_resblk * 8, 512)
            self.up3 = UNetUp(1024, 256)
            self.res3 = nResNet(no_resblk * 8, 512)
            self.up4 = UNetUp(512 + 256, 128)
            self.res4 = nResNet(no_resblk * 4, 256)
            self.up5 = UNetUp(256 + 128, 64)
            self.res5 = nResNet(no_resblk * 2, 128) 
            self.up6 = UNetUp(128 + 64, 64)
            self.res6 = nResNet(no_resblk * 1, 64) 

       
        self.final = nn.Sequential(
                nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
            )
        
        self.lin = nn.Sequential(nn.Linear(self.n_classes, 512 * 4*6))
        self.reduce = nn.Sequential(nn.Conv2d(512+32, 512, 3, stride=1, padding=1))
        self.label_emb = nn.Embedding(self.n_classes,self.n_z)

    def forward(self, dd, noise=None, label=None):
        d1,d2,d3,d4,d5,d6,l7,l8 = dd[0],dd[1],dd[2],dd[3],dd[4],dd[5],dd[6],dd[7]
        #h = self.img_height
        #w = self.img_width
        #self.print(d6)
        if noise is None:
            noise = torch.normal(0, 1,size=(l7.shape[0],self.n_z)).cuda()
        if label is None:
            #label = torch.argmax(l7)
            x = torch.mul(l7, noise)
            #print("no label:",x)
        else:
            #el= torch.nn.functional.one_hot(label, num_classes=self.n_classes)
            el = self.label_emb(label)
            x = torch.mul(el, noise)
            #print("label:",label,x)
        xx = self.lin(x)
        xx = xx.view(d6.shape[0],-1,4,6)
        d6 = torch.cat((d6,xx),1)
        d6 = self.reduce(d6)
        d7 = self.down7(d6)
        #self.print(d6)

        nested = self.nested
        attention = self.attention

        # v1 = self.up1(d7, self.res1(d6))
        # v2 = self.up2(v1, self.res2(d5))
        # v3 = self.up3(v2, self.res3(d4))
        # v4 = self.up4(v3, self.res4(d3))
        # v5 = self.up5(v4, self.res5(d2))
        # v6 = self.up6(v5, self.res6(d1))

        if nested:
            u1 = self.up1(d7, self.res1(d6))
            u2_1 = self.up1(d6, self.res2(d5))
            u2 = self.up2(u1, self.res2_1(u2_1))
            u3_1 = self.up3_1(d5, self.res3(d4))
            u3_2 = self.up3_2(u2_1, self.res3_1(u3_1))
            u3 = self.up3(u2, self.res3_2(u3_2))
            u4_1 = self.up4_1(d4, self.res4(d3))
            u4_2 = self.up4_2(u3_1, self.res4_1(u4_1))
            u4_3 = self.up4_3(u3_2, self.res4_2(u4_2))
            u4 = self.up4(u3, self.res4_3(u4_3))
            u5_1 = self.up5_1(d3, self.res5(d2))
            u5_2 = self.up5_2(u4_1, self.res5_1(u5_1))
            u5_3 = self.up5_3(u4_2, self.res5_2(u5_2))
            u5_4 = self.up5_4(u4_3, self.res5_3(u5_3))
            u5 = self.up5(u4, self.res5_4(u5_4))
            u6_1 = self.up6_1(d2, self.res6(d1))
            u6_2 = self.up6_2(u5_1, self.res6_1(u6_1))
            u6_3 = self.up6_3(u5_2, self.res6_2(u6_2))
            u6_4 = self.up6_4(u5_3, self.res6_3(u6_3))
            u6_5 = self.up6_5(u5_4, self.res6_4(u6_4))
            u6 = self.up6(u5, self.res6_5(u6_5))   
        elif attention:
            v1 = self.up1(d7, (d6))
            v2 = self.up2(v1, (d5))
            v3 = self.up3(v2, (d4))
            v4 = self.up4(v3, (d3))
            v5 = self.up5(v4, (d2))
            v6 = self.up6(v5, (d1))

            if self.lateral==True:
                a1 = self.att1((v1), self.res1(d6))#a1 = self.att1((d6), self.res1(d6))
                u1 = self.up1(d7, self.res1(a1))
                a2 = self.att2((v2), self.res2(d5))#a2 = self.att2((d5), self.res2(d5))
                u2 = self.up2( u1, self.res2(a2))
                a3 = self.att3((v3), self.res3(d4))#a3 = self.att3((d4), self.res3(d4))
                u3 = self.up3(u2, self.res3(a3))
                a4 = self.att4((v4), self.res4(d3))#a4 = self.att4((d3), self.res4(d3))
                u4 = self.up4(u3, self.res4(a4))
                a5 = self.att5((v5), self.res5(d2))#a5 = self.att5((d2), self.res5(d2))
                u5 = self.up5(u4, self.res5(a5))
                a6 = self.att6((v6), self.res6(d1))#a6 = self.att6((d1), self.res6(d1))
                u6 = self.up6(u5, self.res6(a6))  
            else:
                a1 = self.att1(self.res1(d6), (d7))
                u1 = self.up1((d7), self.res1(a1))
                a2 = self.att2(self.res2(d5), (v1))#a2 = self.att2(self.res2(d5), (u1))
                u2 = self.up2((u1), self.res2(a2))
                a3 = self.att3(self.res3(d4), (v2))#a3 = self.att3(self.res3(d4), (u2))
                u3 = self.up3((u2), self.res3(a3))
                a4 = self.att4(self.res4(d3), (v3))#a4 = self.att4(self.res4(d3), (u3))
                u4 = self.up4((u3), self.res4(a4))
                a5 = self.att5(self.res5(d2), (v4))#a5 = self.att5(self.res5(d2), (u4))
                u5 = self.up5((u4), self.res5(a5))
                a6 = self.att6(self.res6(d1), (v5))#a6 = self.att6(self.res6(d1), (u5))
                u6 = self.up6((u5), self.res6(a6))       
        else:
            u1 = self.up1(d7, self.res1(d6))
            u2 = self.up2(u1, self.res2(d5))
            u3 = self.up3(u2, self.res3(d4))
            u4 = self.up4(u3, self.res4(d3))
            u5 = self.up5(u4, self.res5(d2))
            u6 = self.up6(u5, self.res6(d1))  

        
        
        #z0 = gram_matrix(d7)
        #z1 = gram_matrix(d6)
        #z2 = gram_matrix(d5)
        z0 = self.final(v6) if attention else self.final(u6) 
        fout = self.final(u6)
        #z0 = z.view(z.size(0),1,z.size(1),-1)
        #return self.final(u6), z0, eout, l6
        #return fout, z0, z1, z2
        return fout, z0, l7, l8


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class MultiDiscriminator(nn.Module):
    def __init__(self, args):
        super(MultiDiscriminator, self).__init__()
        self.normalize = False#args.normalize
        self.disc_channel = args.disc_channel
        self.relu_act = True#args.relu_act
        self.disc_kernel = args.disc_kernel
        swish_act = False#args.swish_act
        self.name = 'discriminator_C'
        self.dropout = args.dropout
        #self.channels = 3 # z:1 d1:64 d2:128
        
        def discriminator_block(in_filters, out_filters, normalize=self.normalize, relu_act=self.relu_act, dropout=self.dropout):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, self.disc_kernel, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            else:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:    
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
            #if style:
            #    layers.append(nn.Conv2d(out_filters, out_filters, self.disc_kernel, stride=1, padding=1))
            layers.append(nn.Dropout2d(dropout))
            return layers

        channels = args.n_channel
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(self.disc_channel):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1 , 3 , padding=1),
                    #Print(),
                    nn.Flatten(),
                    #Print(),
                    nn.Linear(16*24,1),
                    #Print(),
                    
                ),
            )
        
    
    def forward(self, x):
        # total = sum([m(x) for m in self.models])
        # new_x = x
        x = x
        # print(type(x),type(new_x))
        total = [m(x) for m in self.models]
        total = torch.stack(total)
        return total
        # return self.model(x)
    def compute_out(self,x):
        x = x
        total = sum([m(x) for m in self.models])
        return total
        #return self.model(x)

       
    


    
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


