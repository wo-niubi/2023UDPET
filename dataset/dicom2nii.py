import dicom2nifti
import os
import re
path_2_all_patients = "/data/lyy/PART1"
patients_folders = os.listdir(path_2_all_patients)
path_out_data = "/data/lyy/data/nii"
path_to_save_nifti_file= "pet-nii"

for i, patient in enumerate(patients_folders):
	typefolder=os.listdir(os.path.join(path_2_all_patients, patient))
	os.mkdir(f'{path_out_data}/{i}')
	for n, type in enumerate(typefolder):
		dicom2nifti.dicom_series_to_nifti(
			os.path.join(path_2_all_patients, patient,type),
			os.path.join(path_out_data,str(i), str(type).split(' WB ')[1]))

