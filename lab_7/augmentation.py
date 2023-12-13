import os
import cv2
import numpy as np

def augment_dataset(input_folder, output_folder, rotation_range=(-20, 21, 1), scale_factor=0.8,
                    background_color=(255, 255, 255)):

    os.makedirs(output_folder, exist_ok=True)

    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        img = cv2.imread(image_path)

        for angle in range(rotation_range[0], rotation_range[1], rotation_range[2]):
            rotated_img = augment(img, angle, background_color)
            output_file = f"{os.path.splitext(image_file)[0]}_{angle}"+image_path[image_path.rfind('.'):]
            output_path = os.path.join(output_folder, output_file)
            cv2.imwrite(output_path, rotated_img)


def augment(image, angle,  background_color):

    center = tuple(np.array(image.shape[1::-1]) / 2)

    scale_factor = 1.0 - 0.03 * abs(angle)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_img = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
    return rotated_img


input_folder = 'dataset_1'
output_folder = 'dataset_2'

augment_dataset(input_folder, output_folder)
