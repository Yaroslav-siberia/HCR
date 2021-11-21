'''
визуализация
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(img, aabbs):
    img = ((img + 0.5) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for aabb in aabbs:
        aabb = aabb.enlarge_to_int_grid().as_type(int)
        cv2.rectangle(img, (aabb.xmin, aabb.ymin), (aabb.xmax, aabb.ymax), (255, 0, 255), 2)

    return img

def visualize_and_plot(img, aabbs,delta):
    '''

    :param img:
    :param aabbs: список границ/выделений
    :param delta: отступ, в пикселях. увеличивает рамки вокруг слова на эту величину чтобы все буквы влезли целиком
    и ненароком не обрезать букву с длиным хвастом
    :return:
    '''
    plt.imshow(img, cmap='gray')
    for aabb in aabbs:
        start_point = (int(aabb.xmin-delta), int(aabb.ymin-delta))
        end_point = (int(aabb.xmax+delta), int(aabb.ymax+delta))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0),1)
    return img






