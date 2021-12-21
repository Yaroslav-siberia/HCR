import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
#import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        # проверка на существование директории
        assert data_dir.exists()

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        for line in f:
            # если будут комментарии в файле сопоставления пропустить строки
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ') # файл сопоставления вида *название картинки* *слово на картинке*
            assert len(line_split) >= 2

            file_name_split = line_split[0]
            file_base_name = line_split[0]
            file_name = data_dir / 'img' /  file_base_name

            # если на картинке есть пробелы то все после первого пробела объединяем
            gt_text = ' '.join(line_split[1:])
            chars = chars.union(set(list(gt_text)))

            # составляем списки картинка-слово
            self.samples.append(Sample(gt_text, file_name))
            #print(Sample(gt_text, file_name))

        # разбиваем на части для обучения и валидации: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # составляем списки слов для обучения и валидации
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # обучение
        self.train_set()

        # список всех всех символов
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """возвращает данные для тренировки"""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """возвращает данные для валидации"""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """получение информации об итерации обучения"""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """для выполнения шага итерации"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        '''получение картинки'''
        img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        #print(self.samples[i].file_path)

        return img

    def get_next(self) -> Batch:
        """для выполнения шага итерации"""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
