import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Module, Sequential,Conv3d,Conv2d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, ReLU, Sigmoid
# from model.gausfilter import GausFilter,GaussianKernel
from scipy.ndimage import median_filter
import copy

class Triple(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(Triple, self).__init__()
        self.backbone=Backbone(in_channels, out_channels, features)
        self.conv1=nn.Sequential(ChanelAttention(features,features),
                                nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
                                )
        self.conv2=nn.Sequential(ChanelAttention(features,features),
                                nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
                                )
    def forward(self, x):  
        dec=self.backbone(x)
        information=self.conv1(dec)
        denoise=self.conv2(dec)
        return denoise+information


class Backbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super(Backbone, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.encoder1 = block(self.in_channels, self.features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = block(self.features, self.features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = block(self.features * 2, self.features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = block(self.features * 4, self.features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = block((self.features * 8) , self.features * 4)
        self.upconv2 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = block((self.features * 4) , self.features *2)
        self.upconv1 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = block(self.features * 2, self.features)
    def forward(self, x):  
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        dec3 = self.decoder3(torch.cat((self.upconv3(enc4),enc3),dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3),enc2),dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2),enc1),dim=1))
        return dec1

class Dhead(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        self.layer=nn.Sequential(ChanelAttention(features,features),
                                nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1),
                                )
    def forward(self, x):
        output=self.layer(x)
        return x-output


def block(in_channels, features):
    block = nn.Sequential(nn.Conv3d(in_channels,features,kernel_size=3,padding=1,),
            nn.BatchNorm3d(features,eps=1e-3, affine=True, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(features,features,kernel_size=3,padding=1),
            nn.BatchNorm3d(features,eps=1e-3, affine=True, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
    return block

class Encoder(nn.Module):
    def __init__(self, in_channels=1, features=16):
        super(Encoder, self).__init__()
        self.encoder1 = block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = block(features * 8, features * 16)
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.encoder5(self.pool4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

class Decoder(nn.Module):
    def __init__(self, out_channels=1, features=16):
        super(Decoder, self).__init__()
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = block(features * 2, features)
    def forward(self,enc1,enc2,enc3,enc4,bottleneck):
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

class ChanelAttention(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ChanelAttention, self).__init__()
        self.layer=nn.Sequential(
             nn.AdaptiveAvgPool3d(1),
             nn.Conv3d(in_channels,out_channels,kernel_size=1),
             nn.Sigmoid()
        )
    def forward(self,x):
        map=1+self.layer(x).expand_as(x)
        output=torch.mul(map,x)
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, features=2):
        super(Discriminator, self).__init__()
        self.encoder1 = block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = block(features * 4, features * 8)
        self.layer=nn.Sequential(nn.Linear(1936000*features,4),
                                )
    def forward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc4=torch.flatten(enc4, start_dim=1)
        output= self.layer(enc4)
        return output

class MedianFilter(nn.Module):
    def __init__(self, kernel_size):
        super(MedianFilter, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input):
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        padding = self.kernel_size // 2
        unfolded_input = F.unfold(input, self.kernel_size, padding=padding)

        unfolded_input = unfolded_input.view(input.size(0), input.size(1), -1, self.kernel_size**2)
        median_value, _ = torch.median(unfolded_input, dim=-1)

        output = median_value.view(input.size(0), input.size(1), input.size(2), input.size(3))
        return output




# X=torch.randn((2,1,640,440,440)).to(torch.device("cuda:2"))
# Y=Discriminator().to(torch.device("cuda:2"))
# z=Y(X)
# print(z.shape,z)
