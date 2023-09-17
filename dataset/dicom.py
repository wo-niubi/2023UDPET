# 读取文件夹下的dcm文件
# 读取三维图像体数据
import numpy as np
import torch
from glob import glob
import SimpleITK as sitk
import imageio
from PIL import Image
from pydicom import dcmread
from matplotlib import pyplot
import os
import re



# volread函数接收把目录作为参数 将所有医学数字成像和通信文件汇编为一个numpy的三维数组
def dicom_file2jpg(dicomfile_path,out_path):
    dcm_list =glob(dicomfile_path +'/*.dcm')  #获取该路径下所有的.dcm文件
    dcm_list.sort()
    image = sitk.ReadImage(dcm_list)  #生成3D图像（自动按照世界坐标从小到大排序）
    image_arr = sitk.GetArrayFromImage(image) #将3D图像转换为3D数组

    (z,x,y) = image_arr.shape  # 获得数据shape信息：（长，宽，维度-即切片数量）
    for k in range(z):
        X = dcmread(dcm_list[k])
        X = (X.pixel_array).astype(float)
        scaled_image = (np.maximum(X, 0) / X.max()) * 255.0
        scaled_image = np.uint8(scaled_image)
        scaled_image=torch.tensor(scaled_image)
        final_image = Image.fromarray(np.uint8(scaled_image.numpy()))
        final_image.save(f'{out_path}/{k}.jpg')

path_2_all_patients = "/data/lyy/PART1"
patients_folders = os.listdir(path_2_all_patients)#不同患者
path_out_data = "/data/lyy/data/jpg"

for i, patient in enumerate(patients_folders):
 
    typefolder=os.listdir(os.path.join(path_2_all_patients, patient))#患者下的不同剂量dicom
    if not os.path.exists(f'{path_out_data}/{i}'):
        os.mkdir(f'{path_out_data}/{i}')
    for n, type in enumerate(typefolder):
        dicomfile_path=os.path.join(path_2_all_patients, patient,type)
        print(str(type).split("WB "))
        out_path=os.path.join(path_out_data,str(i), str(type).split("WB ")[1])
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        dicom_file2jpg(dicomfile_path,out_path)