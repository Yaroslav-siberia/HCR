'''
Самая важная часть которая все делает.
'''

import os
import argparse
import torch
from path import Path
import cv2
from DetectingWordsNN.src.dataloader import DataLoaderImgFile
from DetectingWordsNN.src.eval import evaluate
from DetectingWordsNN.src.net import WordDetectorNet
from typing import List
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import namedtuple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# подгружаем нашу нейронную сеть
net = WordDetectorNet()  # получаем архитектуру нашей нейронной сети без весов
#  загружаем веса нашей нейронной сети
net.load_state_dict(torch.load('./DetectingWordsNN/model/weights', map_location=device))
net.eval()  # переводим сеть в режим распознавания (вдруг она была в режиме обучения)
net.to(device)  # отправляем нейронку на устройство которое выбрали

def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png','*.jpeg', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res

def sort_line(detections):
    """упорядочевание строк"""
    return [sorted(detections, key=lambda det: det.xmax / 2)]

def sort_multiline(detections,
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) :
    '''
    кластеризует слова в строки и упорядочевает их на странице
    :param detections: список детектов
    :param max_dist: максимальн жакардово расстояние
    :param min_words_per_line: мин число слов в строке
    :return: список строк с детектами
    '''

    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        res += sort_line(line)
    return res

def _cluster_lines(detections,
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2):
    '''
    клестеризует слова в строки, вернее определяет какие слова стоят на одной строке а какие нет
    :param detections:
    :param max_dist:
    :param min_words_per_line: минимальное число слов в строке
    :return:
    '''
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(num_bboxes):
            a = detections[i]
            b = detections[j]
            if a.ymin > b.ymax or b.ymin > a.ymax:
                continue
            intersection = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
            union = (a.ymax - a.ymin) + (b.ymax - b.ymin) - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = 1 - iou  # Jaccard distance is defined as 1-iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det.ymax / 2 for det in line])
    return res

def line_box(line):
    x_min = 8000
    x_max = 0
    y_min = 8000
    y_max = 0
    for word_idx,det in enumerate(line):
        if det.xmin < x_min:
            x_min = det.xmin
        if det.xmax > x_max:
            x_max = det.xmax
        if det.ymin < y_min:
            y_min = det.ymin
        if det.ymax > y_max:
            y_max = det.ymax
    return x_min, x_max, y_min, y_max

def extract_strings(input_path,output_path, delta):
    '''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)'''

    '''# подгружаем нашу нейронную сеть
    net = WordDetectorNet()  # получаем архитектуру нашей нейронной сети без весов
    #  загружаем веса нашей нейронной сети
    net.load_state_dict(torch.load('../model/weights', map_location=device))
    net.eval()  # переводим сеть в режим распознавания (вдруг она была в режиме обучения)
    net.to(device)  # отправляем нейронку на устройство которое выбрали'''

    # создаем загрузчик изобраний
    loader = DataLoaderImgFile(Path(input_path), net.input_size, device)
    res = evaluate(net, loader, max_aabbs=1000)
    # max_aabbs максимальное число слов на листе. более чем достаточно
    #приступаем к работе !!!!!!!!!!!!!
    # цикл по изображениям и спискам детектов на каждом
    num_colors = 8
    colors = plt.cm.get_cmap('rainbow', num_colors)
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        print(f'Processing {i} image')
        f = loader.get_scale_factor(i) # определяем направление изображения
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs] # применяем это к рамкам
        lines = sort_multiline(aabbs)
        img , img_name = loader.get_original_img_with_name(i) # получаем оригинальное изобра и его имя
        img_name = img_name.split('/')[-1] # выделяем только нахвание
        index = 0
        for line_idx, line in enumerate(lines):
            x_min, x_max, y_min, y_max = line_box(line)
            #img = cv2.rectangle(img,(int(x_min)-delta,int(y_min)-delta),(int(x_max)+delta,int(y_max)+delta),(255, 0, 0),2)
            string = img[int(y_min)-delta:int(y_max)+delta,int(x_min)-delta:int(x_max)+delta]
            file_name = output_path + '/' + img_name.split('.')[0] + '_' + str(index) + '.' + img_name.split('.')[1]
            try:
                cv2.imwrite(file_name, string * 255)
                print(f'write {index} image')
                index += 1
            except:
                print('error with image writing')

def extract_wordsNN(input_path,output_path, delta):
    '''

    :param input_path: директория с изображениями для поиска
    :param output_path: директория с сохраненными словами
    :param delta: отступ от границ детекта в пикселях, сделано чтобы хваосты слов не обрезать
    :return:
    '''
    #проверяем папку с исходными изображениями
    if os.path.isdir(Path(input_path)):
        res=[]
        for ext in ['*.png', '*.jpeg', '*.jpg', '*.bmp']:
            res += Path(input_path).files(ext)
        if len(res)== 0:
            print('Изображений для обработки не обнаружено. проверьте путь')
        else:
            print(f"Найдено {len(res)} изображений для обработки.")
    else:
        print(f'директория {input_path} не существует')
    #проверяем папку ку складываем
    if os.path.isdir(output_path):
        print(f'директория {output_path} уже существует')
    else:
        os.makedirs(output_path)
        print(f'директория {output_path} создана')
    # определяем устройство работы
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # подгружаем нашу нейронную сеть
    net = WordDetectorNet()  # получаем архитектуру нашей нейронной сети без весов
    #  загружаем веса нашей нейронной сети
    net.load_state_dict(torch.load('../model/weights', map_location=device))
    net.eval()  # переводим сеть в режим распознавания (вдруг она была в режиме обучения)
    net.to(device)  # отправляем нейронку на устройство которое выбрали

    # создаем загрузчик изобраний
    loader = DataLoaderImgFile(Path(input_path), net.input_size, device)
    res = evaluate(net, loader, max_aabbs=1000)  #max_aabbs максимальное число слов на листе. более чем достаточно
    #приступаем к работе !!!!!!!!!!!!!
    # цикл по изображениям и спискам детектов на каждом
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        print(f'Processing {i} image')
        f = loader.get_scale_factor(i) # определяем направление изображения
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs] # применяем это к рамкам
        img , img_name = loader.get_original_img_with_name(i) # получаем оригинальное изобра и его имя
        img_name = img_name.split('/')[-1] # выделяем только нахвание

        index=0
        for aabb in aabbs: # вырезаем каждое найденное слово/символ и сохраняем
            word = img[int(aabb.ymin - delta):int(aabb.ymax + delta), int(aabb.xmin - delta):int(aabb.xmax + delta)]
            file_name = output_path+'/' + img_name.split('.')[0] + '_' + str(index) + '.' + img_name.split('.')[1]
            try:
                cv2.imwrite(file_name,word*255)
                print(f'write {index} image')
                index += 1
            except:
                print('error with image writing')
        print(f'all words from image {i} were writen')
        '''img2 = visualize_and_plot(img, aabbs, 5)  # покеазываем как выделили слова
        cv2.imshow('detected words', img2)
        cv2.waitKey(0) '''

DetectItem = namedtuple('DetectItem','img_name, lines')

def extract_words_sorted(input_path):
    detection_list=[]
    # создаем загрузчик изобраний
    loader = DataLoaderImgFile(Path(input_path), net.input_size, device)
    res = evaluate(net, loader, max_aabbs=1000)
    # max_aabbs максимальное число слов на листе. более чем достаточно
    # приступаем к работе !!!!!!!!!!!!!
    # цикл по изображениям и спискам детектов на каждом
    num_colors = 8

    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        print(f'Processing {i} image')
        f = loader.get_scale_factor(i)  # определяем направление изображения
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]  # применяем это к рамкам
        lines = sort_multiline(aabbs)
        img, img_name = loader.get_original_img_with_name(i)  # получаем оригинальное изобра и его имя
        detection_list.append(DetectItem(img_name,lines))
    return detection_list


