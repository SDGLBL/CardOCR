# -*- coding:utf-8 -*-
import os

import PIL.Image as Image
import numpy as np
from keras.layers import Input
from keras.models import Model

from ocr.densenet.lib.Densenet import densenet
from utils.config.dconfig import DFLAG as FLAG
from utils.decode_dict import dict


# 获取解码字符库
def _get_characters():
    characters = dict.alphabet
    characters = characters[0:] + u'卍'
    return characters


# CTC解码函数
def _decode_ypred(ypred, characters=_get_characters(), nclass=FLAG.NCLASS):
    char_list = []
    # 取第二维的最大值，第一维为图片排序
    pred_text = ypred.argmax(axis=2)[0]
    # 使用CTC算法解码
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1]))
                or
                (i > 1 and pred_text[i] == pred_text[i - 2])
        ):
            # 防止识别多次空格
            if len(char_list) > 0 and char_list[-1] == '_' and characters[pred_text[i]] == '_':
                pass
            else:
                char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


# 获取Denesnet+BRNN联合网络，以产生未解码数据
def _get_basemodel(path=FLAG.MODEL_PATH, nclass=FLAG.NCLASS, use_blstm=FLAG.IS_USE_BLSTM):
    # reload(densenet)
    input = Input(shape=(FLAG.IMG_HEIGHT, None, FLAG.IMG_CHANEEL), name='the_input')
    y_pred = densenet(input, nclass, use_blstm)
    basemodel = Model(inputs=input, outputs=y_pred)
    if os.path.exists(path):
        basemodel.load_weights(path)
    else:
        raise RuntimeError('basemodel不存在,请确认model存在。')
    return basemodel


# 通过解码函数与联合网络识别图像
def predict(img, basemodel=_get_basemodel()):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / FLAG.IMG_HEIGHT
    width = int(width / scale)
    # resize输入图像为（FLAG.IMG_HEIGHT，width）
    img = img.resize([width, FLAG.IMG_HEIGHT], Image.ANTIALIAS)
    # 归一化
    img = np.array(img).astype(np.float32) / 255 - 0.5
    # 重整为四维数组
    input = img.reshape((1, FLAG.IMG_HEIGHT, width, 1))
    ypred = basemodel.predict(input)
    # 取结果的前三维
    ypred = ypred[:, :, :]
    # 使用CTC解码
    result = _decode_ypred(ypred)

    return result
