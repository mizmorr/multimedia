import gzip
import os
import shutil
import cv2
from pathlib import Path # pathlib for easy path handling
import pydicom # pydicom to handle dicom files
import matplotlib.pyplot as plt
import numpy as np
import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files

import dicom2nifti


def unpack_single_gzip_in_folder(folder_path):

    files = os.listdir(folder_path)


    gz_files = [file for file in files if file.endswith('.gz')]


    if len(gz_files) == 1:
        gz_file_path = os.path.join(folder_path, gz_files[0])


        output_file_path = os.path.splitext(gz_file_path)[0]

        with gzip.open(gz_file_path, 'rb') as f_in, open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file_path)
        print(f"Распаковано: {gz_file_path} -> {output_file_path}")
    else:
        print("Ошибка: Не удалось определить единственный файл .gz в указанной папке.")


# folder = os.getcwd()+"/100_200_studies"
folder = os.getcwd()
# path  = Path("/home/temporary/work/multimedia/individual_3/100_200_studies/1.2.643.5.1.13.13.12.2.77.8252.15150908151113110911000201110706/1.2.643.5.1.13.13.12.2.77.8252.06110502031105001300030609150005")
# dicom2nifti.convert_directory(path, ".")
# nifti = nib.load("3.nii.gz")
# print(nifti)
# head_mri = nifti.get_fdata()

# tested = head_mri[:,:, 23]
# kernel = np.ones((5,5),np.uint8)
unpack_single_gzip_in_folder(os.path.join(folder))
# tested = cv2.cvtColor(tested,cv2.COLOR_BGR2GRAY)
# print(tested.shape)
# dst = cv2.detailEnhance(tested, sigma_s=10, sigma_r=0.15)

# lab= cv2.cvtColor(tested, cv2.COLOR_BGR2LAB)
# l_channel, a, b = cv2.split(lab)

# # Applying CLAHE to L-channel
# # feel free to try different values for the limit and grid size:
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl = clahe.apply(l_channel)

# # merge the CLAHE enhanced L-channel with the a and b channel
# limg = cv2.merge((cl,a,b))

# # Converting image from LAB Color model to BGR color spcae
# enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# # Stacking the original image with the enhanced image
# result = np.hstack((tested, enhanced_img))

# fig, axis = plt.subplots(1, 1, figsize=(20, 10))

# axis[0][0].imshow(result)
# plt.waitforbuttonpress()
# # slice_counter = 0
# for i in range(3):
#     for j in range(4):
#         axis[i][j].imshow(head_mri[:,:,slice_counter])
#         slice_counter+=1


# for x in os.listdir(folder):
#     current = os.path.join(folder,x)
#     if os.path.isdir(current):
#         nested_current = os.path.join(current, os.listdir(current)[0])
#         # dicom2nifti.convert_directory(nested_current,os.path.join(folder,"result"))
#         print(nested_current)

    # dicom2nifti.convert_directory(os.path.join(folder,x))
    # dicom2nifti.convert_directory(os.path.join("/home/temporary/work/multimedia/individual_3/100_200_studies", x,
    #                                os.listdir(os.path.join(
    #                                                    "/home/temporary/work/multimedia/individual_3/100_200_studies",
    #                                                    x))[0]), os.path.join("/home/temporary/work/multimedia/individual_3/100_200_studies", x))

