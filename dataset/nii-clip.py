import SimpleITK as sitk
import os
import random
import numpy as np

root='/data/lyy/data/uExplorerPART1/'
for i in range(11,14):
    print(i)
    patientpath=root+str(i)
    typefolder=os.listdir(patientpath)
    for n, type in enumerate(typefolder):
        low_image = sitk.ReadImage(patientpath+'/'+str(type))
        low_array = sitk.GetArrayFromImage(low_image)
        # roi_array=low_array[40:650,80:270,40:320]
        roi_array=low_array[250:570]
        roi_image = sitk.GetImageFromArray(roi_array)
        roi_image.SetSpacing(low_image.GetSpacing())
        roi_image.SetDirection(low_image.GetDirection())
        roi_image.SetOrigin(low_image.GetOrigin())
        sitk.WriteImage(roi_image, patientpath+'/'+str(type))
# 230:560,230:580,210:560,215:565,230:580,230:560,260:610
# 260:590,240:560,250:560,250:560,250:570,250:570,250:570