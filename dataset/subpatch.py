import numpy as np
import nibabel as nib
import csv,os,glob
import pandas as pd
from torch.utils.data import DataLoader

def SubPatch(array_shape,size):
    list=[]
    D,H,W= array_shape#array.shape
    stepx = size[0]//2
    stepy = size[1]//2
    stepz = size[2]//2
    patch_depth= size[0]
    patch_height= size[1]
    patch_width= size[2]
    for z in range(0, W, stepz):
        for y in range(0, H, stepy):
            for x in range(0, D, stepx):
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

                xyz_list = [x_min,x_max, y_min,y_max, z_min,z_max]
                list.append(xyz_list)
    patch_list = []
    for i in list:
        if i not in patch_list:
            patch_list.append(i)
    return patch_list                    


root='/data2/lyy/2023uExplorer/'
patch_list=SubPatch((673,360,360),(192,192,192))
print(len(patch_list))

def PrepareData(root,patch_list):
    patient_list=os.listdir(root)
    patient_list.sort(key=lambda x: int(x))

    for patient in patient_list:
        patient_path=root+patient
        if 300>=int(patient):
            print(patient)
            type='train'
            index=1
            array_1_100=nib.load(f'{patient_path}/1-100 dose.nii.gz').get_fdata().transpose(2,0,1)
            # array_1_50=nib.load(f'{patient_path}/1-50 dose.nii.gz').get_fdata().transpose(2,0,1)
            # array_1_20=nib.load(f'{patient_path}/1-20 dose.nii.gz').get_fdata().transpose(2,0,1)
            # array_1_10=nib.load(f'{patient_path}/1-10 dose.nii.gz').get_fdata().transpose(2,0,1)
            # array_1_4=nib.load(f'{patient_path}/1-4 dose.nii.gz').get_fdata().transpose(2,0,1)
            full_array=nib.load(f'{patient_path}/Full_dose.nii.gz').get_fdata().transpose(2,0,1)
            
            if array_1_100.shape==full_array.shape and array_1_100.shape==(673,360,360):
                for location in patch_list:
                    patch_1_100=array_1_100[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
                    # patch_1_50=array_1_50[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
                    # patch_1_20=array_1_20[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
                    # patch_1_10=array_1_10[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
                    # patch_1_4=array_1_4[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
                    patch_full=full_array[location[0]:location[1],location[2]:location[3],location[4]:location[5]]
        
                    np.save(f'/data2/lyy/patch/{type}/1-100/{patient}_{index}', patch_1_100)
                    # np.save(f'/data2/lyy/patch/{type}/1-50/{patient}_{index}', patch_1_50)
                    # np.save(f'/data2/lyy/patch/{type}/1-20/{patient}_{index}', patch_1_20)
                    # np.save(f'/data2/lyy/patch/{type}/1-10/{patient}_{index}', patch_1_10)
                    # np.save(f'/data2/lyy/patch/{type}/1-4/{patient}_{index}', patch_1_4)
                    np.save(f'/data2/lyy/patch/{type}/Full_dose/{patient}_{index}', patch_full)

                    index=index+1
                # if not(full_array.shape==array_1_4.shape==array_1_10.shape==array_1_20.shape==array_1_50.shape==array_1_100.shape):
                #     print(full_array.shape,array_1_4.shape,array_1_10.shape,array_1_20.shape,array_1_50.shape,array_1_100.shape)
                #     print(patient)
            else:
                print('存在尺寸不相同，患者序号为',patient)
        



PrepareData(root,patch_list)


def MakeCSV(type):
    DataPath=f'/data2/lyy/patch/train' 
    low_dose_path=DataPath+f'/{type}/'
    high_dose_path=DataPath+'/Full_dose/'
    patch_list=os.listdir(low_dose_path)
    # patch_list=[x for x in patch_list if int((x).split('_')[0])<=190]
    with open(f'{type}''.csv','w') as file:
        for patch in patch_list:
            # if int(patch.split('_')[0])<250:
                low_dose_patch=low_dose_path+patch
                high_dose_patch=high_dose_path+patch
                dir=[]
                dir.append(low_dose_patch)
                dir.append(high_dose_patch)
                writer = csv.writer(file)
                writer.writerow(dir)
            # else:
            #     print(patch)

MakeCSV('1-100')

def MakeLowCSV():
    dose_name=['1-100 dose.nii.gz','1-50 dose.nii.gz','1-20 dose.nii.gz','1-10 dose.nii.gz','1-4 dose.nii.gz']
    DataPath='/data2/lyy2/2023Quadra2/'
    with open(f'low''.csv','w') as file:
        for i in range(371):
            for n in range(4):
                dir=[]
                dir.append(f'/data2/lyy2/2023Quadra2/{i}/'+dose_name[n])
                dir.append(n)
                writer = csv.writer(file)
                writer.writerow(dir)
            # print(f'/data2/lyy2/2023Quadra2/{i}/'+dose_name[n])


# low=pd.read_csv('/data/lyy/dataset/patch.csv').iloc[:, 0].values
# low1 = [x for x in low if (x).split('_')[-1].split('.npy')[0]==str(1)]
# print(len(low1))

# perm = np.arange(len(low))
# np.random.shuffle(perm)
# low=low[perm]
# high=high[perm]

# low = [x for x in low if (x).split('lowdose/')[-1].split('_')[0]!='10']
# high = [x for x in high if (x).split('highdose/')[-1].split('_')[0]!='10']

# dataset=PatchData(low,high)
# dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

# i=0
# for batch in (dataloader):
#     i=i+1
# print(i)
# for i in range(len(low)):
#     x=(low[i].split('lowdose/')[-1]==high[i].split('highdose/')[-1])
#     print(low[i].split('lowdose/')[-1],high[i].split('highdose/')[-1])
#     if x==False:
#          print(x)
