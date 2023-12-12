import pytesseract as tess
from PIL import Image
import time
import easyocr
import os
import json
import cv2
import numpy as np
from augmentation import augment
class Recognition:

    def __init__(self,annotation_file='annotation.json',):
        self.annotation_file = annotation_file
        self.additional_annotation = 'additional.json'

    def read_annotations(self):
        annotations= list()
        f = open(self.annotation_file)
        data = json.load(f)
        for row in data:
            annotations.append(list(map(list, row.items()))[0])
        return annotations

    # def read_annotations(self, annotation_file):
    #     with open(annotation_file, 'r', encoding='utf-8') as file:
    #         annotations = [line.strip().split(maxsplit=1) for line in file.readlines()]
    #     return annotations


    def straight_recognition(self, img):
        start_time = time.time()
        text = tess.image_to_string(img, lang='rus+eng')
        end_time = time.time()
        res_time = end_time - start_time
        return text.strip(), res_time

    def easyocr_recognition(self, image_path):
        start_time = time.time()
        reader = easyocr.Reader(['en', 'ru'])
        result = reader.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        end_time = time.time()
        elapsed_time = end_time - start_time
        return text.strip(), elapsed_time

    def accuracy(self, expected_text, recognized_text):

        expected_words = set(expected_text.lower().split())
        recognized_words = set(recognized_text.lower().split())
        full_match = expected_text.lower() == recognized_text.lower()

        if len(recognized_words)==0:
            error_percent = 1.0
        else:
            error_percent = len(expected_words.symmetric_difference(recognized_words).intersection(recognized_words))/len(recognized_words)

        return full_match, error_percent

    def start_recognition(self, rec_type, images_folder):
        annotations = self.read_annotations(self.annotation_file)
        results = []
        all_full=0
        general_error=0
        start_time = time.time()

        for annotation in annotations:
            image_name, expected_text = annotation
            image_path = os.path.join(images_folder, image_name)
            img = Image.open(image_path)

            if rec_type == 'straight_recognition':
                recognized_text, elapsed_time = self.straight_recognition(img)
            elif rec_type == 'easyocr_recognition':
                recognized_text, elapsed_time = self.easyocr_recognition(image_path)
            else:
                raise ValueError("Неподдерживаемый тип распознавания")

            full_match, error = self.accuracy(expected_text, recognized_text)


            results.append({
                'Имя файла': image_name,
                'Ожидаемый текст': expected_text,
                'Распознанный текст': recognized_text,
                'Полное совпадение': full_match,
                'Ошибка': error,
                'Время распознавания': elapsed_time
            })
            all_full+=int(full_match)
            general_error+=error

        result_file_path = f'results_1/{rec_type}.json'
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            json.dump(results, result_file, indent=4, ensure_ascii=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        all_full/=len(annotations)
        mean_error = general_error/len(annotations)
        res_string = f"Метод разпознавания:{rec_type}\nПроцент полного совпадения:{all_full*100}%\nОбщая ошибка:{general_error}\nСреднее ошибки:{mean_error}%\nВремя распознования:{elapsed_time}\n--------------\n"
        with open('results_1/general_results.txt','a',encoding='utf-8') as general:
            general.write(res_string)
            general.close()

    def task_3(self,img_name):
            f = open (self.additional_annotation, "r")

            data = json.loads(f.read())
            expected_text = data[img_name]
            background_color=(255, 255, 255)
            parent_folder = img_name[0:img_name.find('.')]+"_results"

            if os.path.isdir(parent_folder):
                os.system(f'rm -rf {parent_folder}')

            images_folder = parent_folder+"/img"
            os.makedirs(parent_folder, exist_ok=True)
            os.makedirs(images_folder, exist_ok=True)
            general_error = 0
            general_full = 0
            results = []
            img = cv2.imread("dataset_1/"+img_name,cv2.IMREAD_ANYCOLOR)
            start_time = time.time()

            for angle in range(-20, 21, 1):
                rotated_img = augment(img, angle, background_color)
                recognized_text, elapsed_time = self.straight_recognition(img)
                full_match, error = self.accuracy(expected_text, recognized_text)

                output_name =  img_name[0:img_name.find('.')]+f'({angle})'+img_name[img_name.find('.'):]
                results.append({
                        'Имя файла': output_name,
                        'Ожидаемый текст': expected_text,
                        'Распознанный текст': recognized_text,
                        'Полное совпадение': full_match,
                        'Ошибка': error,
                        'Время распознавания': elapsed_time
                })

                general_full+=int(full_match)
                general_error+=error
                output_path = f'{images_folder}/{output_name}'
                cv2.imwrite(output_path, rotated_img)

            result_file_path = parent_folder+'/tests.json'
            with open(result_file_path, 'w', encoding='utf-8') as result_file:
                    json.dump(results, result_file, indent=4, ensure_ascii=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            general_full/=41
            mean_error = general_error/41
            res_string = f"--------------------------\nПроцент полного совпадения:{general_full*100}%\nОбщая ошибка:{general_error}\nСреднее ошибки:{mean_error}%\nВремя распознования:{elapsed_time}\n--------------------------\n"
            with open(parent_folder+'/general_results.txt','a',encoding='utf-8') as general:
                general.write(res_string)
                general.close()
            cv2.imshow("original",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


recognizer = Recognition()
# recognizer.start_recognition(rec_type='straight_recognition', annotation_file='annotation.json', images_folder='dataset_1')
# recognizer.start_recognition(rec_type='easyocr_recognition', annotation_file='annotation.json', images_folder='dataset_1')
recognizer.task_3('commit.png')
