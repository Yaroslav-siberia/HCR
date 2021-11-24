import cv2
from path import Path
from typing import List

def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def resizing (input_path, output_path):
    #приводим разрешение изображения к HD (1280х720) подобному
    #
    #
    size = 900
    print(input_path)
    for fn_img in get_img_files(Path(input_path)):

        print(f'Processing file {fn_img}')
        img = cv2.imread(fn_img)
        height, width, channels = img.shape
        if height < size and width < size:
            print('1',fn_img)
            fn_img = fn_img.split('/')[-1]
            file_name = output_path + '/' + fn_img
            print(file_name)
            try:
                cv2.imwrite(file_name, img)
            except:
                print('error with image writing')
            continue
        elif height > width:
            print('2',fn_img)
            f=size/height
            new_heght = int(f*height)
            new_width = int(f*width)
            res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            fn_img = fn_img.split('/')[-1]
            file_name = output_path + '/' + fn_img
            print(file_name)
            try:
                cv2.imwrite(file_name, res_img)
            except:
                print('error with image writing')
        elif width >= height:
            print('3',fn_img)
            f=size/width
            new_heght = int(f*height)
            new_width = int(f*width)
            res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            fn_img = fn_img.split('/')[-1]
            file_name = output_path + '/' + fn_img
            print(file_name)
            try:
                cv2.imwrite(file_name, res_img)
            except:
                print('error with image writing')

resizing ('/home/ysiberia/PycharmProjects/HandWrittenTR/resize_img/input', '/home/ysiberia/PycharmProjects/HandWrittenTR/resize_img/output900')


