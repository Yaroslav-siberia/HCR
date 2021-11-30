import cv2
from path import Path
from typing import List

def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def resizing_img (image, size = 900):
    #приводим разрешение изображения к HD (1280х720) подобному
    #
    #
    img = image.copy()
    height, width, channels = img.shape
    if height < size and width < size:
        # надо увеличить
        # определяем наибольшую сторону и подгоняем ее под размер 900 пикселей, вторую сторону изменяем пропорционально
        if height > width:
            f = size / height
            res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            return res_img
        elif width >= height:
            f = size / width
            res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            return res_img
    # уменьшение
    elif height > width:
        f=size/height
        res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return res_img
    elif width >= height:
        f=size/width
        res_img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return res_img



