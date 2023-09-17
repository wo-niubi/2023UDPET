import os
import csv
import pandas as pd

# root=r'/data/lyy/data/ROIProcess'

# partList=os.listdir(root)
# with open('train''.csv','w') as file:
#     for part in partList:
#         # 设置旧文件名（就是路径+文件名）
#         part_dir=os.path.join(root,part)
#         patientList=os.listdir(part_dir)
#         for patient in patientList:
#             patient_dir=os.path.join(part_dir,patient)
#             # doseList=os.listdir(patient_dir)
#             # for dose in doseList:
#             #     dose_name=os.path.join(patient_dir,dose)
           
                
#             writer = csv.writer(file)
#             writer.writerow([f"{patient_dir}"])
# patient=pd.read_csv('/home/ubuntu/pet-recon/data/test.csv').iloc[:,0].values
# print(patient.shape)
# path=os.path.join(str(patient[0]),'D100'+'nii.gz')
# print(path)
# print(len(patient))

path=r'/data2/lyy2/2023Quadra'

file_list=os.listdir(path)
file_list.sort(key=lambda x: int(x.split('_')[-1]))
# print((file_list))

def rename(file_list):
    idx=1
    for file in file_list:
        # 原文件夹名
        old_folder_name = path+'/'+file
        # 新文件夹名
        new_folder_name = path+'/'+str(idx)
        # 重命名文件夹
        os.renames(old_folder_name, new_folder_name)
        print(file)
        idx=idx+1
rename(file_list)