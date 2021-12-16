import os

def startRename(dir):
    import os
    i = 1
    for file in os.listdir(dir):
        ext=file.split('.')[-1]
        os.rename(f'{dir}/{file}', f'{dir}/{i}.{ext}')
        i = i + 1


startRename('/home/ysiberia/Документы/GitHub/HCR/data/input')