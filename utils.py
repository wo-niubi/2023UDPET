import numpy as np
import torch
from torch import nn
import os
from dataset.mydata import PatchData
from torch.utils.data import DataLoader
import SimpleITK as sitk
import nibabel as nib
from model.unet import UNet3d
from model.triple import Net,Discriminator
from model.loss import L1Loss,MSELoss,GradientLoss,calc_psnr
from model.stylegan import Dip
from skimage.metrics import structural_similarity as compare_ssim
from evaluation import compute_ssim,compute_mse,compute_nrmse,compute_psnr,compute_mae
import pandas as pd
import os , time ,csv

def train():
    G=Net().to(torch.device("cuda:1"))
    G.load_state_dict(torch.load(('/data2/lyy/b.pth')))
    G_optimizer= torch.optim.AdamW(G.parameters(), lr= 0.01)
    for i in range(10):
        x=torch.randn((2,1,4,4,4)).to(torch.device("cuda:1"))
        loss=torch.mean(G(x)-1)
        G_optimizer.zero_grad()
        loss.backward()
        G_optimizer.step()
    torch.save(G.state_dict(), '/data2/lyy/d.pth')

# train()

def show_pth():
    content1 = torch.load(('/data2/lyy2/run/triple3-huber5my.pth'),map_location=torch.device("cuda:2"))
    # content1 = torch.load(('/data2/lyy/run/triple3-l1denoise.pth'),map_location=torch.device("cuda:2"))
    content2 = torch.load(('/data2/lyy2/run/triple3-huber5denoise.pth'),map_location=torch.device("cuda:2"))
    key_list1=content1.keys()
    key_list2=content2.keys()
    for key in key_list2:
        if not (content1[key]==content2[key]).all():
            print(key)

# show_pth()

# for i in range(280,317):
#     array=nib.load(f'/data2/lyy/2023uExplorer/{i}/1-100 dose.nii.gz').get_fdata()
#     print(array.shape,i)