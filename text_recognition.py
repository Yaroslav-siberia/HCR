import cv2
from path import Path
import os
from DetectingWordsNN.src.infer import get_img_files, extract_words_sorted, line_box
from HTR.src.main import recognition
import matplotlib.pyplot as plt
import time

def extract_text(input_path,output_path,delta):
    if os.path.isdir(Path(input_path)):
        if len(get_img_files(Path(input_path))) == 0:
            print('Изображений для обработки не обнаружено. проверьте путь')
        else:
            print(f"Найдено {len(get_img_files(Path(input_path)))} изображений для обработки.")
    else:
        print(f'директория {input_path} не существует')
    # проверяем папку ку складываем
    if os.path.isdir(output_path):
        print(f'директория {output_path} уже существует')
    else:
        os.makedirs(output_path)
        print(f'директория {output_path} создана')
    # получили список файлов
    files = get_img_files(input_path)
    for_recognition = extract_words_sorted(input_path)
    for DetectItem in for_recognition:
        text=''
        img = cv2.imread(DetectItem.img_name, cv2.IMREAD_GRAYSCALE)
        for line in DetectItem.lines:
            '''x_min, x_max, y_min, y_max = line_box(line)
            crop_string = img[int(y_min)-delta:int(y_max)+delta,int(x_min)-delta:int(x_max)+delta]
            plt.imshow(crop_string, cmap='gray')
            plt.show()
            time.sleep(5)
            try:
                characters = recognition(crop_string)
                print(characters)
                text += ' ' + characters[0]
            except:
                print('Cant recognize')'''
            for word in line:
                print(word)
                crop_word = img[int(word.ymin)-delta:int(word.ymax)+delta,int(word.xmin)-delta:int(word.xmax)+delta]
                plt.imshow(crop_word, cmap='gray')
                plt.show()
                time.sleep(5)
                # следующая строка нужна для бинаризации на случай если используется модель обученная на бинаризированных картинках
                crop_word = cv2.threshold(crop_word, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                try:
                    characters = recognition(crop_word)
                    print(characters)
                    text+=' '+characters[0]
                except:
                    print('Cant recognize')
            text += ' \n '
        f_name = output_path+'/'+DetectItem.img_name.split('/')[-1]+'.txt'
        with open(f_name,'a') as my_file:
            my_file.write(text)
            my_file.close()

extract_text('/home/ysiberia/Документы/GitHub/HCR/data/input','/home/ysiberia/Документы/GitHub/HCR/data/texts',10)