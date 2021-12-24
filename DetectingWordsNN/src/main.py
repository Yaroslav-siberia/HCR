from infer import extract_strings

from typing import List
import os
from path import Path
import cv2
from resizing import resizing_img


def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png','*.jpeg', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def extract_words(input_path,output_path):
    if os.path.isdir(Path(input_path)):
        if len(get_img_files(Path(input_path)))== 0:
            print('Изображений для обработки не обнаружено. проверьте путь')
        else:
            print(f"Найдено {len(get_img_files(Path(input_path)))} изображений для обработки.")
    else:
        print(f'директория {input_path} не существует')
    #
    middle_path = input_path.split('/')[0:-1]
    middle_path.append('middle')
    middle_path = '/'.join(x for x in middle_path)
    if os.path.isdir(middle_path):
        print(f'директория директория промежуточных результатов {middle_path} уже существует')
    else:
        os.makedirs(middle_path)
        print(f'директория директория промежуточных результатов {middle_path} создана')
    #проверяем папку ку складываем
    if os.path.isdir(output_path):
        print(f'директория {output_path} уже существует')
    else:
        os.makedirs(output_path)
        print(f'директория {output_path} создана')

    files = get_img_files(input_path)
    for fn_img in files:
        image = cv2.imread(fn_img,cv2.COLOR_BGR2RGB)
        #уменьшаем изображение, 900 размер большей стороны в пикселях
        img = resizing_img(image,1300)
        # сканирование изображения
        #ans, img = scaner(img)
        f_name=os.path.basename(fn_img)
        # сохраняем промежуточные результаты
        cv2.imwrite(str(os.path.join(middle_path, f_name)),img)
    print('start ')
    # вызываем функцию поиска слов нейронной сетью
    #extract_wordsNN(middle_path,output_path,5)
    extract_strings(middle_path, output_path, 5)

#extract_words('/home/ysiberia/Документы/GitHub/HCR/data/input', '/home/ysiberia/Документы/GitHub/HCR/data/strings')
'''
Надо обратить внимание на функцию _cluster_lines в файле infer 
параметр min_words_per_line определяет количество слов в строке. то есть если в строке только одно слово - оно будет пропущена.
этот параметр нужен для борьбы с ложными детектами нейронки и для борьбы с людьми, которые конец строки "задираю" ну и вообще пишут волной
'''



