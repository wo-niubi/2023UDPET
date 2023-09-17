# import dicom2nifti
import os
import re
import nibabel as nib

path_2_all_patients = r'/data/lyy/data/11'
patients_folders = os.listdir(path_2_all_patients)
path_out_data = r'/data/lyy/data/uExplorer17'
dose_dir=['D20','D2','NORMAL']

for i, patient in enumerate(patients_folders):
	if not patient.startswith('.'):
		typefolder=os.listdir(os.path.join(path_2_all_patients, patient))
		if not os.path.isdir(f'{path_out_data}/{i}'):
			os.mkdir(f'{path_out_data}/{i}')

		for n, type in enumerate(typefolder):
			
			if type.split('.nii')[0] in dose_dir:
				nii_path=os.path.join(path_2_all_patients, patient,type)
				gz_path=os.path.join(path_out_data,str(i), type+'.gz')
				img = nib.load(nii_path)
				img_affine = img.affine
				img = img.get_fdata()
				nib.Nifti1Image(img, img_affine).to_filename(gz_path)


# img = nib.load(r"X:\nii\uExplorer17\72\D2.nii")
# img_affine = img.affine
# img = img.get_fdata()

# nib.Nifti1Image(img, img_affine).to_filename(r"X:\nii\uExplorer17\72\D2.nii.gz")

# img2 = nib.load(r"X:\nii\uExplorer17\72\D2.nii.gz")
# img2_affine = img2.affine
# # img2 = img2.get_fdata()

import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D



# print(img2)
# print(img2.header['db_name'])  # 输出nii的头文件
# width, height, queue = img2.dataobj.shape
# # OrthoSlicer3D(img2.dataobj).show()

# num = 1
# for i in range(0, queue, 10):
#     img_arr = img2.dataobj[:, :, i]
#     # plt.subplot(5, 4, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1
#
# plt.show()
# # if (img==img2).all():
# #     print('yes')
# #
# # if (img_affine==img2_affine).all():
# #     print('yes')