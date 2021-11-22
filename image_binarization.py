import cv2
import os
from path import Path

def binarize_image(file_name):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # load image
    img = cv2.imread(file_name)
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # бинаризация изображения
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #cv2.imshow("THRESH", thresh)  # показать получившееся изображение
    #cv2.waitKey(0)  # закрыть окно с изображением по нажатию любой кнопки

    # в строке 19 адаптивная бинаризация. тут надо менять последние 2 параметра
    # 17, 8 определяют размер площади на которых проходит адаптисная бинаризация
    # метод сделан на случай если на изобржении не равномерная яркость (засвет скана, пятна и т.д.).
    #thresh4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 8)
    #cv2.imshow("THRESH4", thresh4)
    #cv2.waitKey(0)
    return thresh

def binarize_images_from_dir(input_path, output_path):
    '''
    :param input_path: директория с изобранриями для бинаризации
    :param output_path: Директория куда складыватся бинаризированные изображения
    :return:
    '''
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(input_path).files(ext)
    if len(res) == 0:
        print('Изображений для обработки не обнаружено. проверьте путь')
    else:
        print(f"Найдено {len(res)} изображений для обработки.")
    # проверяем папку куда складываем
    if os.path.isdir(output_path):
        print(f'директория {output_path} уже существует')
    else:
        os.makedirs(output_path)
        print(f'директория {output_path} создана')
    for file in res:
        bin = binarize_image(file)
        bin_name = output_path+'/'+file.split('/')[-1]
        cv2.imwrite(bin_name, bin)
    print('Бинаризация завершена')

binarize_images_from_dir('./DetectingWordsNN/data/output','./binarized')