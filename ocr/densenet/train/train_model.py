# -*- coding:utf-8 -*-
# 用于训练模型
import os
import sys
from imp import reload

sys.path.append('../../..')
import numpy as np

import tensorflow as tf

from PIL import Image
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model

from ocr.densenet.lib.Densenet import densenet
from utils.config.dconfig import DFLAG as CONFIG

# 参数设置
IMG_HEIGHT = CONFIG.IMG_HEIGHT
NCLASS = CONFIG.NCLASS
IS_USE_BLSTM = CONFIG.IS_USE_BLSTM
BATCH_SIZE = CONFIG.BATCH_SIZE
MAX_LABEL_LENGTH = CONFIG.MAX_LABEL_LENGTH
IMG_SHAPE = CONFIG.IMG_SHAPE
SHUFFLE_SIZE = CONFIG.SHUFFLE_SIZE


# CTC损失lambda损失函数
def _ctc_func_lambda(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


# Densenet+**blstm**+CTC Model获取函数
def _get_union_model(img_h=IMG_HEIGHT, nclass=NCLASS, is_use_bllistm=IS_USE_BLSTM):
    # 输入层
    input = Input(shape=(img_h, None, 1), name='the_input')
    # 获取y_pred
    y_pred = densenet(input, nclass, is_use_bllistm)
    # densenet
    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()
    # label层
    labels = Input(shape=[None], dtype='float32', name='the_labels')
    # 输入ctc层的长度 ps：图像长度//8
    input_length = Input(shape=[1], dtype='int64', name='input_length')
    # 输入ctc层的label长度
    label_length = Input(shape=[1], dtype='int64', name='label_length')
    # 联合损失函数
    union_loss = Lambda(_ctc_func_lambda, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    # 联合模型定义
    union_model = Model(inputs=[input, labels, input_length, label_length], outputs=union_loss)
    # 定义联合模型的compile方式
    union_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    return basemodel, union_model


# 获取session（多线程且调用GPU加速）
def _get_session(use_gpu=False, gpu_fraction=0.7):
    if (use_gpu == False):
        return tf.Session()
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


# TODO(lijie) imagesize需要感觉训练集的图像格式进行设置
def gen(data_file, image_path, batchsize=BATCH_SIZE,
        maxlabellength=CONFIG.MAX_LABEL_LENGTH, imagesize=(60, 256)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while True:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            # 将下划线替换为空格
            str = image_label[j]
            label_length[i] = len(str)

            if (len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            # TODO(lijie) 此处 k -1 则可训练不识别空格
            labels[i, :len(str)] = [int(k) for k in str]

        inputs = {
            'the_input': x,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def get_file_size(path):
    count = 0
    thefile = open(path)
    while True:
        buffer = thefile.read(1024 * 8192)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close()
    return count


# 模型训练函数
def train_model():
    # 设定训练会话，使用GPU加速
    K.set_session(_get_session(use_gpu=CONFIG.IS_USE_GPU))
    reload(densenet)
    # 获取稠密神经网络和联合神经网络
    basemodel, union_model = _get_union_model()
    # 获取预训练模型
    modelPath = 'model/weights_densenet-01-1.14.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('成功读取预训练模型，继续迁移训练')
    # 模型保存checkpoint ps：只保存有weights，使用需要densenet.py的代码载入参数
    checkpoint = ModelCheckpoint(filepath='./model/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    # 学习率设置函数
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./model/logs', write_graph=True)
    # 获取训练集迭代器
    train_loader = gen('./datasets/data_train.txt', './datasets/images')
    # 获取验证集迭代器
    vali_loader = gen('./datasets/data_vali.txt', './datasets/images')
    print("----------------------开始训练模型--------------------------")
    union_model.fit_generator(train_loader,
                              steps_per_epoch=get_file_size(path='./datasets/data_train.txt') // BATCH_SIZE,
                              epochs=10,
                              initial_epoch=0,
                              validation_data=vali_loader,
                              validation_steps=get_file_size(path='./datasets/data_vali.txt') // BATCH_SIZE,
                              callbacks=[checkpoint, earlystop, changelr, tensorboard])


if __name__ == '__main__':
    train_model()
    # 训练完之后自动停机，云端GPU服务器太贵了，不停的话第二天醒来就会泪流不止
    os.system('zip -r  model.zip /input/neuralNetworks/densenet/train/model/')
    os.system('cp /input/neuralNetworks/densenet/train/model.zip /data')
    os.system('rm /input/neuralNetworks/densenet/train/model.zip')
    os.system(
        'mv /input/neuralNetworks/densenet/train/model /input/neuralNetworks/densenet/train/model_before')
    os.system('mkdir /input/neuralNetworks/densenet/train/model')
    os.system('shutdown')

# NUM_DICT={'0':1, '1':2, '2':3, '3':4,
#               '4':5,'5':6,'6':7,'7':8,
#               '8':9,'9':10, }
#
# #按照字典将原本的数据解码成可用于神经网络使用的数据
# def _process_label_batch(labels, label_batch, raw_images, images_batch, input_length, label_length, img_shape=IMG_SHAPE, batch_size=BATCH_SIZE, dict=NUM_DICT):
#     for i in range(len(labels)):
#         label=[]
#         for k in labels[i]:
#             if(k!='_'):
#                 label.append(dict[k])
#         if (len(labels[i]) <= 0):
#             print("len < 0")
#         images_batch[i]=raw_images[i]
#         label_batch[i,:len(label)]=[num-1 for num in label]
#         input_length[i]=img_shape[1] // 8
#         label_length[i]=len(label)
#
# #数据集迭代函数(获取原始图像与label进行处理再提供给神经网络进行训练)
# def get_iter(file_path,shuffle_size=SHUFFLE_SIZE,batch_size=BATCH_SIZE,image_shape=IMG_SHAPE,maxLength=MAX_LABEL_LENGTH):
#     _iter=tfr_iter.get_tfrecords_iterator(file_path, shuffle_size, batch_size)
#     img_batch = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float)
#     label_batch = np.ones([batch_size, maxLength])*10000
#     input_length = np.zeros([batch_size, 1])
#     label_length = np.zeros([batch_size, 1])
#     for raw_images,labels in _iter:
#         #将bytes的字符串转换为str方便后续处理
#         for i in range(len(labels)):
#             labels[i] = str(labels[i])[2:2+maxLength]
#         #处理从tfrecord读出的数据，整理为输入
#         _process_label_batch(labels, label_batch, raw_images, img_batch, input_length, label_length)
#         #将输入按照神经网络输入层的标签名字定义为字典
#         inputs = {
#                   'the_input': img_batch,
#                   'the_labels': label_batch,
#                   'input_length': input_length,
#                   'label_length': label_length,
#         }
#         outputs = {'ctc': np.zeros([batch_size])}
#         yield inputs,outputs
