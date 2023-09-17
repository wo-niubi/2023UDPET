import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
from model.unet import UNet3d
from model.cGAN import cgan_G
from model.loss import L1Loss,MSELoss,AdapLoss,calc_metric
from model.stylegan import Dip
from model.redcnn import RedCnn
from model.triple import Triple3,Triple
from dataset.mydata import PatchData
from evaluation import compute_ssim,compute_mse,compute_nrmse,compute_psnr,compute_mae
from stop import EarlyStopping
from test import Test,inference
import os , time ,csv
import pandas as pd
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import sys

epoch=2
way='triple'
LR=2e-4
size=(96,96,96)
train_bs=8
valid_bs=16
norm='zs'
dose='1-100'
lossname='huber_10'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    def __init__(self,epoch,way,device,LR,size,norm,lossname,dose): 
        self.epoch=epoch
        self.way=way
        self.device=device
        self.LR=LR
        self.size=size
        self.norm=norm
        self.loss_name=lossname
        self.loss_criterion=self.get_loss(lossname)
        self.dose=dose
        self.early_stopping = EarlyStopping(patience=3, verbose=True,model_path=f'/data2/lyy/run/{way}-{dose}-{lossname}96.pth')
        
    def dataloder(self,csv,norm,batchsize,shuffle=False,num_workers=True):
        if num_workers:
            nw = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])
        else:
            nw=0
        image,label=csv.iloc[:, 0].values,csv.iloc[:, 1].values,
        dataset=PatchData(image,label,norm)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batchsize,drop_last=True,pin_memory=True,num_workers=nw)
        return dataloader

    def get_model(self,way):
        if way=='unet':
            G=UNet3d(1,1,16)
        elif way=='dip':
            G=Dip(init_features=8,in_channels=1, out_channels=1)
        elif way=='cgan':
            G=cgan_G()
        elif way=='redcnn':
            G=RedCnn()
        elif way=='triple':
            G=Triple()
        elif way=='triple3':
            G=Triple3()
            # G.load_state_dict(torch.load((f'/data2/lyy2/run/triple3-{self.loss_name}denoise.pth')))
        return G
    
    def get_loss(self,loss_name):
        if loss_name=='l1':
            L=L1Loss()
        elif loss_name.split('_')[0]=='huber':
            L=torch.nn.HuberLoss(reduction='mean', delta=int(loss_name.split('_')[1]))
        elif loss_name=='mse':
            L=MSELoss()
        return L

    def run(self,train_csv,valid_csv,train_bs,valid_bs):
        start_time=time.time() 
        self.H = {"train_loss": [],  "valdation_loss": []}
        self.train_loader = self.dataloder(train_csv,self.norm,train_bs,shuffle=True)
        # self.valid_loader = self.dataloder(valid_csv,self.norm,valid_bs)
        self.G=self.get_model(self.way).to(self.device)
        self.G_optimizer= torch.optim.AdamW(self.G.parameters(), lr= self.LR)
        self.scheduler_G=torch.optim.lr_scheduler.StepLR(self.G_optimizer, 1, 0.1,verbose=True)
        self.should_stop=False
        for n in range(self.epoch):
            if self.should_stop:
                break  
            else:                
                train_loss_G=self.train(n)
        end_time=time.time()
        print('epoch_time:',end_time-start_time)
        return self.should_stop

    def train(self,n):
        self.G.train()
        sum_loss = torch.zeros(1).to(device)
        sum_psnr = torch.zeros(1).to(device)
        sum_mse = torch.zeros(1).to(device)
        sum_mae = torch.zeros(1).to(device)
        loop = tqdm(self.train_loader, file=sys.stdout)
        itration=len(self.train_loader)
        for idx,batch in enumerate(loop):
            x = batch['image']
            real = batch['label']

            mean = batch['mean']
            std = batch['std']
            x, real = x.to(self.device), real.to(self.device)
            fake = self.predict(x,is_training=True)
            G_loss_L1 = self.loss_criterion(fake,real)

            G_loss=G_loss_L1
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()
            psnr, mse, mae = calc_metric(fake, real, mean, std)

            sum_psnr+=psnr.detach()
            sum_loss+=G_loss.detach()
            sum_mse+=mse.detach()
            sum_mae+=mae.detach()
            loop.desc = "[train epoch {}] loss: {:.3f} psnr: {:.3f} mse: {:.3f} mae: {:.3f}".format(
                n, sum_loss.item() / (idx+1), sum_psnr.item() / (idx+1), sum_mse.item() / (idx+1), sum_mae.item() / (idx+1))
            
            if (idx % (itration//20)== 0 and idx>0 and n>0) or (idx % (itration//20)==0 and idx>itration//2==0):
                ValidPsnr= self.validate(n)
                print('验证集平均PSNR:%.4f 当前学习率：%.6f' %(ValidPsnr,self.G_optimizer.state_dict()['param_groups'][0]['lr']),self.device)
                self.early_stopping(ValidPsnr, self.G)
                if self.early_stopping.early_stop:
                    self.scheduler_G.step()
                    self.early_stopping.early_stop=False               
            if self.G_optimizer.state_dict()['param_groups'][0]['lr']<=2.1e-6:
                self.should_stop=True
                print('earling stop=',idx)
                return sum_loss.item()
        return sum_loss.item()

    def validate(self,n):
        with torch.no_grad():
            self.G.eval()
            sum_psnr = np.zeros(1)
            sum_mse = np.zeros(1)
            sum_mae = np.zeros(1)
            loop2 = tqdm(range(301,317), file=sys.stdout)
            for idx,k in enumerate(loop2):
                low_dose=nib.load(f'/data2/lyy/2023uExplorer/{k}/{self.dose} dose.nii.gz').get_fdata().transpose(2,0,1)
                full_dose=nib.load(f'/data2/lyy/2023uExplorer/{k}/Full_dose.nii.gz').get_fdata().transpose(2,0,1)
                gen_img=inference(low_dose,self.G,self.device,self.size,0.5,self.norm)

                psnr=(compute_psnr(full_dose,gen_img))
                mse=(compute_mse(full_dose,gen_img))
                mae=(compute_mae(full_dose,gen_img))
                sum_psnr+=psnr
                sum_mse+=mse
                sum_mae+=mae
                loop2.desc = "[valid epoch {}] psnr: {:.3f} mse: {:.3f} mae: {:.3f}".format(
                    n, sum_psnr.item() / (idx+1), sum_mse.item() / (idx+1), sum_mae.item() / (idx+1))
            self.G.train()
            return sum_psnr.item()/10 

    def predict(self,img,is_training):
            img = img.contiguous()
            output= self.G(img)
            if not is_training:
                output = output.detach()
            return output


train_csv=pd.read_csv(f'/data2/lyy/dataset/{dose}2.csv')
valid_csv=pd.read_csv(f'/data2/lyy/dataset/{dose}2.csv')

print('begin training')
main=Train(epoch,way,device,LR,size,norm,lossname,dose)
x=main.run(train_csv,valid_csv,train_bs,valid_bs)
print(x)
# print('begin testing')
# Test(f'{way}-{lossname}','/data2/lyy2/2023Quadra/',device,norm)



