import numpy as np
import SimpleITK as sitk
import random
import torch
import nibabel as nib
import csv,os,glob
import pandas as pd

def DicomSeriesReader(dicom_dir):
    """
     read a DICOM series
    :param dicom_dir:input dicom path
    :return: sitk image,series_tags
    """
    # Read the original series. First obtain the series file names using the
    # image series reader.
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        print("ERROR: given directory \"" + dicom_dir + "\" does not contain a DICOM series.")
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()
    tags_to_copy = list(series_reader.GetMetaDataKeys(0))
    tags_to_copy.remove("0020|0013")  # remove instance number
    series_tags = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0, k)]
    return image, series_tags

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            # print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            # print("files:", files)
            return files

dose1_2 = '1-2 dose'
dose1_4 = '1-4 dose'
dose1_10 = '1-10 dose'
dose1_20 = '1-20 dose'
dose1_50 = '1-50 dose'
dose1_100 = '1-100 dose'
dose_full = 'Full_dose'

def GetROIDataU(flag=1):
    src_data_path = r'/data2/lyy/zip_of_uExplorer'
    out_data_roi_path = r'/data2/lyy/2023uExplorer'
    data_dirs_path = file_name_path(src_data_path)
    # print(data_dirs_path)
    dir_file = 0
    print(len(data_dirs_path))

    for dir in range(0, len(data_dirs_path)):
        print(dir)
        # number58，sub_dirs: ['2.886 x 600 WB D2', '2.886 x 600 WB DRF_10', '2.886 x 600 WB DRF_100',
        # '2.886 x 600 WB DRF_20', '2.886 x 600 WB DRF_4', '2.886 x 600 WB DRF_50', '2.886 x 600 WB normal']
        data_dir_path = src_data_path + '/' + data_dirs_path[dir]
        # print(data_dir_path)
        image_dirs_path = file_name_path(data_dir_path)
        for idx,file in enumerate(image_dirs_path):
            tmp=file.split('WB ')[-1]
            if tmp in ['1-2 dose','D2','DRF_2']:
                dose1_2_name=file
            elif tmp in ['1-4 dose','D4','DRF_4']:
                dose1_4_name=file
            elif tmp in ['1-10 dose','D10','DRF_10']:
                dose1_10_name=file
            elif tmp in ['1-20 dose','D20','DRF_20']:
                dose1_20_name=file
            elif tmp in ['1-50 dose','D50','DRF_50']:
                dose1_50_name=file
            elif tmp in ['1-100 dose','D100','DRF_100']:
                dose1_100_name=file
            elif tmp in ['Full_dose','normal','NORMAL']:
                dosefull_name=file
            else:
                print('error',data_dir_path)
        
        # 读取全部序列图像

        dose1_2_image_dir = data_dir_path + '/' + dose1_2_name
        dose1_4_image_dir = data_dir_path + '/' + dose1_4_name
        dose1_10_image_dir = data_dir_path + '/' + dose1_10_name
        dose1_20_image_dir = data_dir_path + '/' + dose1_20_name
        dose1_50_image_dir = data_dir_path + '/' + dose1_50_name
        dose1_100_image_dir = data_dir_path + '/' + dose1_100_name
        dosefull_image_dir = data_dir_path + '/' + dosefull_name

        # dose1_2_image, _ = DicomSeriesReader(dose1_2_image_dir)
        dose1_4_image, _ = DicomSeriesReader(dose1_4_image_dir)
        dose1_10_image, _ = DicomSeriesReader(dose1_10_image_dir)
        dose1_20_image, _ = DicomSeriesReader(dose1_20_image_dir)
        dose1_50_image, _ = DicomSeriesReader(dose1_50_image_dir)
        dose1_100_image, _ = DicomSeriesReader(dose1_100_image_dir)
        dosefull_image, _ = DicomSeriesReader(dosefull_image_dir)
        
        # dose1_2_array = sitk.GetArrayFromImage(dose1_2_image)
        dose1_4_array = sitk.GetArrayFromImage(dose1_4_image)
        dose1_10_array = sitk.GetArrayFromImage(dose1_10_image)
        dose1_20_array = sitk.GetArrayFromImage(dose1_20_image)
        dose1_50_array = sitk.GetArrayFromImage(dose1_50_image)
        dose1_100_array = sitk.GetArrayFromImage(dose1_100_image)
        dosefull_array = sitk.GetArrayFromImage(dosefull_image)

        out_data_roi_dir = out_data_roi_path + '/'  + str(dir_file)
        if not os.path.exists(out_data_roi_dir):
            os.makedirs(out_data_roi_dir)

        # roi_dose1_2_array = dose1_2_array
        # roi_dose1_2_image = sitk.GetImageFromArray(roi_dose1_2_array)
        # roi_dose1_2_image.SetSpacing(dose1_2_image.GetSpacing())
        # roi_dose1_2_image.SetDirection(dose1_2_image.GetDirection())
        # roi_dose1_2_image.SetOrigin(dose1_2_image.GetOrigin())
        # sitk.WriteImage(roi_dose1_2_image, out_data_roi_dir + '/' + dose1_2 + '.nii.gz')

        roi_dose1_4_array = dose1_4_array
        roi_dose1_4_image = sitk.GetImageFromArray(roi_dose1_4_array)
        roi_dose1_4_image.SetSpacing(dose1_4_image.GetSpacing())
        roi_dose1_4_image.SetDirection(dose1_4_image.GetDirection())
        roi_dose1_4_image.SetOrigin(dose1_4_image.GetOrigin())
        sitk.WriteImage(roi_dose1_4_image, out_data_roi_dir + '/' + dose1_4 + '.nii.gz')

        roi_dose1_10_array = dose1_10_array
        roi_dose1_10_image = sitk.GetImageFromArray(roi_dose1_10_array)
        roi_dose1_10_image.SetSpacing(dose1_10_image.GetSpacing())
        roi_dose1_10_image.SetDirection(dose1_10_image.GetDirection())
        roi_dose1_10_image.SetOrigin(dose1_10_image.GetOrigin())
        sitk.WriteImage(roi_dose1_10_image, out_data_roi_dir + '/' + dose1_10 + '.nii.gz')

        roi_dose1_20_array = dose1_20_array
        roi_dose1_20_image = sitk.GetImageFromArray(roi_dose1_20_array)
        roi_dose1_20_image.SetSpacing(dose1_20_image.GetSpacing())
        roi_dose1_20_image.SetDirection(dose1_20_image.GetDirection())
        roi_dose1_20_image.SetOrigin(dose1_20_image.GetOrigin())
        sitk.WriteImage(roi_dose1_20_image, out_data_roi_dir + '/' + dose1_20 + '.nii.gz')

        roi_dose1_50_array = dose1_50_array
        roi_dose1_50_image = sitk.GetImageFromArray(roi_dose1_50_array)
        roi_dose1_50_image.SetSpacing(dose1_50_image.GetSpacing())
        roi_dose1_50_image.SetDirection(dose1_50_image.GetDirection())
        roi_dose1_50_image.SetOrigin(dose1_50_image.GetOrigin())
        sitk.WriteImage(roi_dose1_50_image, out_data_roi_dir + '/' + dose1_50 + '.nii.gz')

        roi_dose1_100_array = dose1_100_array
        roi_dose1_100_image = sitk.GetImageFromArray(roi_dose1_100_array)
        roi_dose1_100_image.SetSpacing(dose1_100_image.GetSpacing())
        roi_dose1_100_image.SetDirection(dose1_100_image.GetDirection())
        roi_dose1_100_image.SetOrigin(dose1_100_image.GetOrigin())
        sitk.WriteImage(roi_dose1_100_image, out_data_roi_dir + '/' + dose1_100 + '.nii.gz')

        roi_dosefull_array = dosefull_array
        roi_dosefull_image = sitk.GetImageFromArray(roi_dosefull_array)
        roi_dosefull_image.SetSpacing(dosefull_image.GetSpacing())
        roi_dosefull_image.SetDirection(dosefull_image.GetDirection())
        roi_dosefull_image.SetOrigin(dosefull_image.GetOrigin())
        sitk.WriteImage(roi_dosefull_image, out_data_roi_dir + '/' + dose_full + '.nii.gz')
        dir_file = dir_file + 1

# GetROIDataU()

# dose1_100_image,_=DicomSeriesReader('/data2/lyy/zip_of_uExplorer/Anonymous_ANO_20220224_1703593_111906/2.886 x 600 WB D100')
# array1=sitk.GetArrayFromImage(dose1_100_image)
# array2=nib.load('/data2/lyy/2023uExplorer/142/1-100 dose.nii.gz').get_fdata().transpose(2,0,1)
# print((array1.shape,array2.shape))
# print((array1==array2).all())


def check_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print("文件夹不存在")
  

    # 获取文件夹中的文件列表
    file_list = os.listdir(folder_path)

    # 检查文件列表是否为空
    if len(file_list) == 0:
        print("文件夹为空",folder_path)
    

# 调用函数并传入文件夹路径
# root='/data2/lyy/zip_of_uExplorer/'
# for patient in os.listdir(root):
#     for dose in  os.listdir(root+patient):
#         folder_path = root+patient+'/'+dose
#         check_files_in_folder(folder_path)


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
patch_list=SubPatch((670,360,360),(192,192,192))


def PrepareData(root,patch_list):
    patient_list=os.listdir(root)
    patient_list.sort(key=lambda x: int(x))

    for patient in patient_list:
        print(patient)
        patient_path=root+patient
        if int(patient)<=280:
            type='train'
        # elif 99<int(patient)<=109:
        #     type='valid'
        else:
            break
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
        
# PrepareData(root,patch_list)
