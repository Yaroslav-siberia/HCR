import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from word_detector import detect, prepare_img, sort_multiline


def get_img_files(data_dir: Path) -> List[Path]:
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def main():
    '''
    для демонстрации
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('./input'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=50)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')
        img = cv2.imread(fn_img)
        height, width, channels = img.shape
        # load image and process it
        img = prepare_img(img, height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

        plt.show()

#if __name__ == '__main__':
#    main()

def extract_words(input_path,output_path,
                  delta = 5,
                  kernel_size=25,
                  sigma = 11,
                  theta = 7,
                  min_area = 500):
    '''
    img_height = 2000 теперь адаптивно
    :param input_path: директория с изображениями для поиска
    :param output_path: директория с сохраненными словами
    :param kernel_size: размер ядра фильтра/свертки.
    :param sigma: Стандартное отклонение функции Гаусса, используемой для ядра фильтра.
    :param theta: Приблизительное соотношение ширины и высоты слов, функция фильтра искажается этим фактором
                    Этот параметр порой необходимо настраивать на месте.
    :param min_area: минималная площадь слова - чтобы отсеять ложные детекты (мелкие) но можно тога потерять знаки
                        препинания
    :return:
    '''
    for fn_img in get_img_files(Path(input_path)):
        print(f'Processing file {fn_img}')
        img = cv2.imread(fn_img)
        height, width, channels = img.shape
        # load image and process it
        prep_img = prepare_img(img, height)
        detections = detect(prep_img,
                            kernel_size=kernel_size,
                            sigma=sigma,
                            theta=theta,
                            min_area=min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        # plot results
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                fn_img=fn_img.split('/')[-1]
                word = img[int(det.bbox.y-delta):int(det.bbox.y + det.bbox.h+delta),
                       int(det.bbox.x-delta):int(det.bbox.x + det.bbox.w+delta)]
                file_name = output_path + '/' + fn_img.split('.')[0] + '_' + 's' + str(line_idx) + 'w' + \
                            str(word_idx) + '.' + fn_img.split('.')[1]
                try:
                    cv2.imwrite(file_name, word )
                except:
                    print('error with image writing')

                #тут закоментирована часть по выведению красивого графика. его надо раскомментировать
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

        plt.show()

extract_words('/home/ysiberia/PycharmProjects/HandWrittenTR/DetectingWords/input','/home/ysiberia/PycharmProjects/HandWrittenTR/DetectingWords/output')