# Handwriten text recognition dataset
https://drive.google.com/file/d/1yYpe0iKnrZ9d4q5fmT09h4tAJudFyzig/view?usp=sharing


# HCR
contains Word Detectors and HandWritten Character Recognition for Russian Language
Для обучения:
В файле HTR/src/main закомментировать функцию recognition и 2 строки перед ней с созданием объектов decoder_type,model
расскомментировать вызов функции start_train. Проверить настройки в функции start_train.
Запустить HTR/src/main

# Внимание!
Данный репозиторий содержит несколько связанных проектов потому requirements для всех один

# resizing
Изменение разрешения изображения, уменьшает если наибольшая сторона изображения превышает пороговое значение

# DetectingWordsNN
Детектирование слов рукописного текста на изображении с использованием нейронных сетей
Более подробное описание и инструкция находятся внутри DetectingWordsNN

# DetectingWords
Детектирование слов рукописного текста на изображении с использованием инструментов Opencv(морфология, бинаризация)
Более подробное описание и инструкция находятся внутри DetectingWords

# Запуск
Для распознавания запустить скрипт text_recognition.py
 