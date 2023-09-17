import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from model.unet import UNet3d
from model.loss import L1Loss,MSELoss,GradientLoss,AdapLoss,BerhuLoss
from model.stylegan import Dip
from model.cGAN import cgan_G,cgan_D
from model.ourmodel import Our
from model.stylegan import StyleDip
from model.triple import Triple3,Triple
from evaluation import compute_ssim,compute_mse,compute_nrmse,compute_psnr,compute_mae
import os , time ,csv,sys
import pandas as pd
from monai.inferers import sliding_window_inference
from tqdm import tqdm

# def inference(input,model,device):
#     input=input[np.newaxis,np.newaxis,:,:,:]
#     input=torch.tensor(input).type(torch.FloatTensor).to(device)
#     pred=sliding_window_inference(input, (192, 96, 96), 8, model,overlap=0.5,mode="gaussian",sigma_scale=0.05)
#     pred=pred.cpu().numpy().squeeze()
#     return pred

def inference(array,model,device,size,overlap,norm):
    D,H,W= (array).shape

    stepx = (size[0] //2)
    stepy = (size[1] //2)
    stepz = (size[2] //2)
    patch_depth= size[0]
    patch_height= size[1]
    patch_width= size[2]
    out_mask = np.zeros((D,H,W))
    out_mask_weight = np.zeros((D,H,W))

    for z in range(0, W-stepz, stepz):
        for y in range(0, H-stepy, stepy):
            for x in range(0, D-stepx, stepx):
                x_min = x
                x_max = x_min + patch_depth
                if x_max > D:
                    x_max = D
                    x_min = D - patch_depth
                y_min = y
                y_max = y_min + patch_height
                if y_max > H:
                    y_max = H
                    y_min = H - patch_height
                z_min = z
                z_max = z_min + patch_width
                if z_max > W:
                    z_max = W
                    z_min = W - patch_width
                patch_xs = array[x_min:x_max, y_min:y_max, z_min:z_max]

                if norm=='zs':
                    mean = np.mean(patch_xs)
                    std = np.std(patch_xs)+(1e-5)
                    patch_xs=(patch_xs-mean)/(std)
                elif norm=='01':
                    min = np.min(patch_xs)
                    max = np.max(patch_xs)+1e-5
                    patch_xs = (patch_xs-min)/(max-min)
                patch_xs=torch.tensor(patch_xs).contiguous().float().unsqueeze(0).unsqueeze(0).to(device)
                predictresult = model(patch_xs)
                predictresult =predictresult.squeeze(0).squeeze(0).detach().cpu().numpy()
                if norm == 'zs':
                    predictresult=predictresult* (std) + mean
                elif norm == '01':
                    predictresult=predictresult*(max-min)+min

                out_mask[x_min:x_max, y_min:y_max, z_min:z_max] = out_mask[
                                                                            x_min:x_max, 
                                                                            y_min:y_max, 
                                                                            z_min:z_max] + predictresult.copy()
                out_mask_weight[x_min:x_max, y_min:y_max, z_min:z_max] = out_mask_weight[
                                                                            x_min:x_max,
                                                                            y_min:y_max,
                                                                            z_min:z_max] + 1.
    out_mask = out_mask / out_mask_weight
    # out_mask = np.around(out_mask)
    out_mask[out_mask < 0] = 0
    return out_mask


root='/data2/lyy2/2023Quadra/'
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_path='triple3-huber5my'
norm='zs'

def Test(model_path,root,device,norm):

    if not os.path.exists(f'/data2/lyy2/fake/{model_path}'):
        os.makedirs(f'/data2/lyy2/fake/{model_path}')

    if model_path.split('-')[0]=='dip':
        model=Dip(init_features=8,in_channels=1, out_channels=1)
    elif model_path.split('-')[0]=='unet':
        model=UNet3d(1,1,16)
    elif model_path.split('-')[0]=='cgan':
        model=cgan_G()
    elif model_path.split('-')[0]=='our':
        model=Our()
    elif model_path.split('-')[0]=='stylegan':
        model=StyleDip(8,1,1)
    elif model_path.split('-')[0]=='triple':
        model=Triple()
    elif model_path.split('-')[0]=='triple2':
        model=Triple2()
    elif model_path.split('-')[0]=='triple3':
        model=Triple3()
    model.load_state_dict(torch.load(('/data2/lyy2/run/'+f'{model_path}'+'.pth'),map_location=device))
    model.to(device).eval()

    nrmse_dose=[]
    psnr_dose=[]
    ssim_dose=[]
    mse_list=[]
    mae_list=[]
    patient_list=os.listdir(root)
    patient_list.sort(key=lambda x: int(x))
    for patient in patient_list:  
        if 150>=int(patient)>=110:
            patient_path=root+patient
            array_1_100=nib.load(f'{patient_path}/1-100 dose.nii.gz')
            full_array=nib.load(f'{patient_path}/Full_dose.nii.gz')
            img_affine = array_1_100.affine
            array_1_100=array_1_100.get_fdata().transpose(2,0,1)
            full_array=full_array.get_fdata().transpose(2,0,1)

            pred=inference(array_1_100,model,device,(192,96,96),0.5,norm)

            # pred=array_1_100
  
            nrmse=(compute_nrmse(full_array,pred))
            psnr=(compute_psnr(full_array,pred))
            ssim=(compute_ssim(full_array,pred))
            nrmse_dose.append(nrmse)
            psnr_dose.append(psnr)
            ssim_dose.append(ssim)

            pred=pred.transpose(1,2,0) 
            image = nib.Nifti1Image(pred,img_affine)
            nib.save(image, f'/data2/lyy2/fake/{model_path}/{patient}-{model_path}.nii.gz')
            # nib.save(image, f'/data2/lyy2/fake/information/{patient}-information.nii.gz')
            print(patient,nrmse,psnr,ssim)


    f = open(f"csv/{model_path}.csv", 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(nrmse_dose)
    writer.writerow(psnr_dose)
    writer.writerow(ssim_dose)
    f.close()

# Test(model_path,root,device,norm)

