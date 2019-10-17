# -*- coding:utf-8 -*-


import logging
import os
import random
import sys
import threading
import time

sys.path.append('../../../..')
from PIL import ImageFile
from utils.decode_dict.dict import alphabet as DICT
from utils.image_tools.tools import DataAugmentation

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 将文件名字转换为数字向量准备写入txt
def processLabel(label):
    number_list = []
    splite_index = label.find('-')
    label = label[splite_index + 1:-5]
    for char in label:
        number_list.append(DICT.find(char))
    result = ''
    for number in number_list:
        result += str(number) + ' '
    return result


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception:
        print(Exception)


lock1 = threading.Lock()
lock2 = threading.Lock()


def imageOps(func_name, image, des_path, file_name, write_train_txt, write_vali_txt, times=1000):
    funcMap = {
        "randomRotation": DataAugmentation.randomRotation,
        "randomCrop": DataAugmentation.randomCrop,
        "randomColor": DataAugmentation.randomColor,
        "randomGaussian": DataAugmentation.randomGaussian
    }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
    if random.randint(1, 10) == 1:
        for _i in range(0, times, 1):
            new_image = funcMap[func_name](image)
            name = func_name + str(_i) + '-' + file_name
            DataAugmentation.saveImage(new_image, os.path.join(des_path, name))
            with lock1:
                write_vali_txt.write(name + ' ' + processLabel(name) + '\n')
    else:
        for _i in range(0, times, 1):
            new_image = funcMap[func_name](image)
            name = func_name + str(_i) + '-' + file_name
            DataAugmentation.saveImage(new_image, os.path.join(des_path, name))
            with lock2:
                write_train_txt.write(name + ' ' + processLabel(name) + '\n')
    print(threading.current_thread().getName() + '完成生成。')


opsList = {"randomRotation", "randomColor"}


def threadOPS(path, new_path, txt_train_path, txt_vali_path):
    """
    多线程处理事务
    :return:
    """
    write_train_txt = open(txt_train_path, 'w')
    write_vali_txt = open(txt_vali_path, 'w')
    if os.path.isdir(path):
        img_names = os.listdir(path)
    else:
        img_names = [path]
    for img_name in img_names:
        print(img_name)
        tmp_img_name = os.path.join(path, img_name)
        if os.path.isdir(tmp_img_name):
            if makeDir(os.path.join(new_path, img_name)) != -1:
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
            else:
                print('create new dir failure')
                return -1
                # os.removedirs(tmp_img_name)
        elif tmp_img_name.split('.')[1] != "DS_Store":
            # 读取文件并进行操作
            image = DataAugmentation.openImage(tmp_img_name)
            threadImage = [0] * 5
            _index = 0
            for ops_name in opsList:
                threadImage[_index] = threading.Thread(target=imageOps,
                                                       args=(ops_name, image, new_path, img_name, write_train_txt,
                                                             write_vali_txt))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.2)


if __name__ == '__main__':
    threadOPS('./raw_images', './images', 'data_train.txt', 'data_vali.txt')
