'''
Самая важная часть которая все делает.
'''

import os
import argparse
import torch
from path import Path
import cv2
from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import visualize_and_plot


def main():
    parser = argparse.ArgumentParser()
    # поределяем на каком устройстве будет работать сеть cuda - видеокарта
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    net = WordDetectorNet()  # получаем архитектуру нашей нейронной сети без весов
    #  загружаем веса нашей нейронной сети
    net.load_state_dict(torch.load('../model/weights', map_location=args.device))
    net.eval()  # переводим сеть в режим распознавания (вдруг она была в режиме обучения)
    #print(net)  # на случай если интересно посмотреть ее архитектуру
    net.to(args.device)  # отправляем нейронку на устройство которое выбрали

    #
    loader = DataLoaderImgFile(Path('../data/input'), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)
    print(res.batch_imgs)
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        img = loader.get_original_img(i)
        print(i)
        cv2.imshow('input', img)
        cv2.waitKey(0)
        img=visualize_and_plot(img, aabbs,5)
        cv2.imshow('input',img)
        cv2.waitKey(0)
        cv2.imwrite(f'./image-{i}.png', img*255)


#if __name__ == '__main__':
#    main()



def extract_words(input_path,output_path, delta):
    '''

    :param input_path: директория с изображениями для поиска
    :param output_path: директория с сохраненными словами
    :param delta: отступ от границ детекта в пикселях, сделано чтобы хваосты слов не обрезать
    :return:
    '''
    #проверяем папку с исходными изображениями
    if os.path.isdir(Path(input_path)):
        if len(Path(input_path).files('*.jpg'))== 0:
            print('Изображений для обработки не обнаружено. проверьте путь')
        else:
            print(f"Найдено {len(Path(input_path).files('*.jpg'))} изображений для обработки.")
    else:
        print(f'директория {input_path} не существует')
    #проверяем папку ку складываем
    if os.path.isdir(output_path):
        print(f'директория {output_path} уже существует')
    else:
        os.makedirs(output_path)
        print(f'директория {output_path} создана')

    #определяем устройство работы
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #подгружаем нашу нейронную сеть
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
            except:
                print('error with image writing')
            index+=1
        img2 = visualize_and_plot(img, aabbs, 5)  # покеазываем как выделили слова
        cv2.imshow('detected words', img2)
        cv2.waitKey(0)

extract_words('../data/input', '../data/output', 5)

