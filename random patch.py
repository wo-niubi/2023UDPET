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

def extract_random_patches(data1,data2, patch_size, num_patches):
    # 从三维data随机切分num_patches个大小为patch_size的patch
    patches1 = []
    patches2 = []
    # patch保存在patches列表中
    max_idxs = np.array(data1.shape) - patch_size
    for _ in range(num_patches):
        start_idx = np.random.randint(max_idxs)
        patch1 = data1[start_idx[0]:start_idx[0]+patch_size,
                     start_idx[1]:start_idx[1]+patch_size,
                     start_idx[2]:start_idx[2]+patch_size]
        patches1.append(patch1)
        patch2 = data2[start_idx[0]:start_idx[0] + patch_size,
                 start_idx[1]:start_idx[1] + patch_size,
                 start_idx[2]:start_idx[2] + patch_size]
        patches2.append(patch2)
    return patches1,patches2


class RandomPatchData(Dataset):
    # 从niigz图像文件中随机切分出指定大小的patch
    def __init__(self, images,labels,patchsize, num_patches):  # （images,labels为低剂量和全剂量niigz文件的地址，patchsize：切分patch的尺寸，num_patches：一个batch切分多少patch）

        self.images = images
        self.labels = labels
        self.patchsize = patchsize
        self.num_patches = num_patches
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #读取一个患者的低剂量和全剂量niigz文件
        imagepath = self.images[index]
        image = nib.load(imagepath).get_fdata()
        labelpath = self.labels[index]
        label = nib.load(labelpath).get_fdata()

        eps = 1e-5
        #z-s归一化
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + eps)
        label = (label - mean) / (std + eps)

        #从image, label中随机切分num_patches个大小为patch_size的patch
        image_patches,label_patches=extract_random_patches(image, label, patch_size, num_patches)
        #把切分出的patch堆叠在一起 从N个（patch_size,patch_size,patch_size）->（N，patch_size,patch_size,patch_size）
        stacked_image_patches = np.stack(image_patches, axis=0)
        stacked_label_patches = np.stack(label_patches, axis=0)

        # transform ndarray to tensor
        mean_tensor = torch.as_tensor(min).float()
        std_tensor = torch.as_tensor(max + eps).float()
        images_tensor = torch.as_tensor(image).float().
        label_tensor = torch.as_tensor(label).float()

        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}





#示例
# csv=pd.read_csv('/data2/lyy/dataset/train.csv')
# dataset=RandomPatchData(csv.iloc[:, 0].values,csv.iloc[:, 1].values,patchsize=192, num_patches=4)
#batch-size设置为1，每次读取一个患者的低剂量和全剂量图像，速度会更快一些，以下代码都为该情况
# dataloader = DataLoader(dataset, batch_size=1)
# for batch in dataloader:
#     print(' ')
#输出的images_tensor或label_tensor的shape应该为（1， num_patches， patchsize, patchsize, patchsize,）
#为了对应（Batch,Channel,D,H,W)的形式，需要把image_tensor和label_tensor转换成（num_patches，1， patchsize, patchsize, patchsize)的形状，
#可以用image_tensor_change=image_tensor.permute(1, 0, 2, 3, 4)实现
