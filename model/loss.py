import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

class L1Loss(nn.Module):
    """
    MSELoss
    """

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, y_pred_logits, y_true):
        bs = y_true.size(0)
        num_classes = y_pred_logits.size(1)
        y_pred_logits = y_pred_logits.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        l1 = self.loss(y_pred_logits, y_true)
        return l1

class MSELoss(nn.Module):
    """
    MSELoss
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, y_pred_logits, y_true):
        bs = y_true.size(0)
        num_classes = y_pred_logits.size(1)
        y_pred_logits = y_pred_logits.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        return self.loss(y_pred_logits, y_true)

class AdapLoss(nn.Module):
    def __init__(self):
        super(AdapLoss, self).__init__()
        self.mseloss = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()
        self.huberloss=torch.nn.HuberLoss(reduction='mean', delta=5.0)

    def forward(self, y_pred_logits, y_true):
        loss_list=[]
        bs = y_true.size(0)
        num_classes = y_pred_logits.size(1)
        y_pred_logits = y_pred_logits.float().contiguous()#.view(bs, num_classes, -1)
        y_true = y_true.float().contiguous()#.view(bs, num_classes, -1)
        for i in range(bs):
            pred=y_pred_logits[i]
            true=y_true[i]
            L2=self.mseloss(pred,true)
            L1=self.l1loss(pred,true)
            if L2<=1:
                loss=L2
            elif L2>1 and L2<L1*L1: #?
                loss=L2
            else:
                loss=self.huberloss(pred,true)
            loss_list.append(loss)
        loss_list=torch.stack(loss_list)
        return torch.mean(loss_list)

class GradientLoss(nn.Module):

    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: 预测的三维tensor,形状为[batch_size, depth, height, width]
        target: 目标的三维tensor,形状为[batch_size, depth, height, width]
        """
        # 计算预测tensor在深度、高度和宽度方向的梯度
        dx = torch.abs(pred[:,:,:,:-1] - pred[:,:,:,1:])
        dy = torch.abs(pred[:,:,:-1,:] - pred[:,:,1:,:])
        dz = torch.abs(pred[:,:-1,:,:] - pred[:,1:,:,:])
        
        # 计算目标tensor在深度、高度和宽度方向的梯度
        dx_target = torch.abs(target[:,:,:,:-1] - target[:,:,:,1:])
        dy_target = torch.abs(target[:,:,:-1,:] - target[:,:,1:,:])
        dz_target = torch.abs(target[:,:-1,:,:] - target[:,1:,:,:])
        
        # 求取差值的绝对值
        dx_loss = torch.abs(dx - dx_target)
        dy_loss = torch.abs(dy - dy_target)
        dz_loss = torch.abs(dz - dz_target)
        
        # 在batch维度上求和,得到最终loss
        return torch.sum(dx_loss) + torch.sum(dy_loss) + torch.sum(dz_loss)

class BerhuLoss(nn.Module):
    def __init__(self,threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_pred_logits, y_true):
        y_pred_logits = y_pred_logits.float().contiguous()#.view(bs, num_classes, -1)
        y_true = y_true.float().contiguous()#.view(bs, num_classes, -1)
        residual=torch.abs(y_pred_logits-y_true)
        # c=self.threshold*torch.max(residual).detach().cpu().numpy()
        c=self.threshold
        part1 = -F.threshold(-residual, -c, 0.)
        part2 = F.threshold(residual**2 - c**2, 0., -c**2.) + c**2
        part2 = part2 / (2.*c)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

def calc_psnr(input: Tensor, target: Tensor, mean: Tensor, std: Tensor):
    """Peak Signal to Noise Ratio
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)"""
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    eps = 1e-7
    psnr = 0
    for batch in range(num):
        mse = torch.mean((input[batch]* std[batch] - target[batch]* std[batch]) ** 2)
        max = torch.max(target[batch] * std[batch] + mean[batch])
        psnr += 10 * torch.log10(max ** 2 / mse + eps)
    return psnr / num

def calc_metric(input: Tensor, target: Tensor, mean: Tensor, std: Tensor):
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    eps = 1e-7
    psnr = 0
    mse_sum=0
    mae_sum=0
    for batch in range(num):
        mae = torch.mean(torch.abs(input[batch]* std[batch] - target[batch]* std[batch]))
        mse = torch.mean((input[batch]* std[batch] - target[batch]* std[batch]) ** 2)
        max = torch.max(target[batch] * std[batch] + mean[batch])
        psnr += 10 * torch.log10(max ** 2 / mse + eps)
        mse_sum+=mse
        mae_sum+=mae
    return psnr / num, mse_sum / num, mae_sum / num



class PredLoss(nn.Module):
    def __init__(self):
        super(PredLoss, self).__init__()
        self.mseloss = torch.nn.MSELoss()
        self.L1loss = torch.nn.L1Loss()

    def forward(self, pred,true,input):  
        residual1=true-input
        residual2=pred-input
        
        residual1[residual1>0]=1
        residual1[residual1<0]=-1

        residual2[residual2>0]=1
        residual2[residual2<0]=-1

        mse_loss=self.mseloss(residual1,residual2)
        L1_loss=self.L1loss(pred,true)
        # print(mse_loss,L1_loss)
        loss=mse_loss+L1_loss
        return loss

class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.L1loss = torch.nn.L1Loss()
    def forward(self, pred,true):  
        residual=pred-true
        residual[residual<0]=0
        loss=torch.mean(residual)
        return loss

def cross_entropy_loss(y_pred,y_true):

    epsilon = 1e-7  # 添加一个小的数以防止对数运算中的除零错误
    y_pred = torch.clip(y_pred, epsilon, 1.0 - epsilon)  # 将预测概率限制在一个接近0或1的范围内，避免取对数时出现无穷大或无穷小
    
    # 计算交叉熵损失
    loss = -torch.sum(y_true * torch.log(y_pred)) / len(y_true)
    
    return loss   




# x=torch.randn((2,2))
# y=torch.randn((2,2))
# loss=PredLoss()
# print(loss(x,y,x))