import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import re
import matplotlib.pyplot as plt
import pandas as pd

class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        self.c=c
        assert (h, w, d) > crop_sz
        self.img_sz  = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)
        
    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        # print(slice_hwd)
        return self._crop(x, self.c,*slice_hwd)
        
    @staticmethod
    def _get_slice(sz, crop_sz):
        try : 
            lower_bound = torch.randint(sz-crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except: 
            return (None, None)
    
    @staticmethod
    def _crop(x, c,slice_h, slice_w, slice_d):
        normalization=x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
        mean = torch.mean(normalization[0])
        std = torch.std(normalization[0])
        for i in range(c):
            normalization[i]=(normalization[i]-mean)/(std+(1e-5))
        return normalization

class MyData(Dataset):
    def __init__(self, image,label,norm=True):
        # 将传入的根目录文件夹和该文件夹下标签文件夹的路径名赋值给类中自身的属性
        self.image = image
        self.label = label
        self.norm = norm
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        imagepath = self.image[index]
        image = np.load(imagepath)
        labelpath = self.label[index]
        label = np.load(labelpath)
        eps = 1e-5

        if self.norm=='zs':
            mean = image.mean()
            std = image.std()
            
            image = (image - mean) / (std + eps)
            label = (label - mean) / (std + eps)
            noise_image = image + awgn(label, -2)

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + eps).float()
        image = torch.as_tensor(image).float().unsqueeze(0)
        label = torch.as_tensor(label).float().unsqueeze(0)
        noise_image = torch.as_tensor(noise_image).float().unsqueeze(0)

        return {'image': image, 'label': label, 'noise_image': noise_image, 'mean': mean_tensor, 'std': std_tensor}

class PatchData(Dataset):
    def __init__(self, images,labels,norm=True):
        # 将传入的根目录文件夹和该文件夹下标签文件夹的路径名赋值给类中自身的属性
        self.images = images
        self.labels = labels
        self.norm = norm
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        imagepath = self.images[index]
        image = np.load(imagepath)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        eps = 1e-5

        if self.norm=='zs':
            mean = image.mean()
            std = image.std()
            
            image = (image - mean) / (std + eps)
            label = (label - mean) / (std + eps)
            mean_tensor = torch.as_tensor(mean).float()
            std_tensor = torch.as_tensor(std + eps).float()
        elif self.norm=='01':
            min,max=np.min(image),np.max(image)
            image=(image-min)/(max-min+ eps)
            label=(label-min)/(max-min+ eps)
            mean_tensor = torch.as_tensor(min).float()
            std_tensor = torch.as_tensor(max + eps).float()
        
        images_tensor = torch.as_tensor(image).float().unsqueeze(0)  # transform ndarray to tensor
        label_tensor = torch.as_tensor(label).float().unsqueeze(0)

        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}

class NoiseData(Dataset):
    def __init__(self, images,labels,norm,snr):
        # 将传入的根目录文件夹和该文件夹下标签文件夹的路径名赋值给类中自身的属性
        self.images = images
        self.labels = labels
        self.norm = norm
        self.snr=snr
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        labelpath = self.images[index]
        label = np.load(labelpath)
       

        mean = label.mean()
        std = label.std()
        if self.norm=='zs':
            label = (label - mean) / (std + 1e-5)
        elif self.norm=='01':
            label=(label-np.min(label))/(np.max(label)-np.mean(label)+1e-5)

        # noise = np.random.normal(0, 0.05, (192,96,96))
        # noise = np.random.poisson(10, (192,96,96))
        # noise = np.random.rayleigh(label, (192,96,96))
        noise = awgn(label, self.snr)
        image=label+noise
        
        images_tensor = torch.as_tensor(image).float().unsqueeze(0)  
        label_tensor = torch.as_tensor(label).float().unsqueeze(0)

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + 1e-5).float()
        return {'image': images_tensor,  'label': label_tensor,'mean': mean_tensor, 'std': std_tensor}

def awgn(x, snr):
    '''
    加入高斯白噪声 
    :x: 原始信号
    :snr: 信噪比
    '''
    noise = np.random.randn(192,96,96)
    noise = noise-np.mean(noise)
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / np.size(x)
    npower = xpower / snr
    noise=noise * np.sqrt(npower) / np.std(noise)
    return noise

class LowData(Dataset):
    def __init__(self, image,label):
        # 将传入的根目录文件夹和该文件夹下标签文件夹的路径名赋值给类中自身的属性
        self.image = image
        self.label = label
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        imagepath = self.image[index]
        labelpath = self.label[index]
        image = nib.load(imagepath).get_fdata()
        label = torch.as_tensor(labelpath)
        
        mean = image.mean()
        std = image.std()
        
        image = (image - mean) / (std + 1e-7)
        image = torch.as_tensor(image).float().unsqueeze(0)

        return {'image': image, 'label': label}


# csv=pd.read_csv('/data2/lyy/dataset/train.csv')
# dataset=NoiseData(csv.iloc[:, 0].values,csv.iloc[:, 1].values,True)
# dataloader = DataLoader(dataset, batch_size=3)
# for batch in dataloader:
#     print(' ')