import os

from utils.decode_dict.dict import alphabet


class DFLAG:
    SHUFFLE_SIZE = 8
    # BATCHSIZE
    BATCH_SIZE = 4
    # label最长长度
    MAX_LABEL_LENGTH = 24
    # 分类的种类数+1（blank算一类）
    NCLASS = len(alphabet) + 1
    # 图像格式（训练集）需要自行根据数据集修改
    IMG_SHAPE = (60, None, 1)
    # 图像深度（通道数）
    IMG_CHANEEL = IMG_SHAPE[2]
    # 图像宽度
    IMG_WIDTH = IMG_SHAPE[1]
    # 图像高度
    IMG_HEIGHT = IMG_SHAPE[0]
    # 预测使用的模型路径，相对位置是从demo.py出发
    MODEL_PATH = os.path.abspath(
        '..') + '/ocr/densenet/train/model/' + 'weights_densenet-03-0.63.h5'
    # 训练模型是否添加双向循环神经网络（默认无须，加入后提升效果有限但会大幅增加训练时间也容易梯度爆炸）
    IS_USE_BLSTM = False
    # 是否使用GPU
    IS_USE_GPU = True
    # 基础数据集图像数目
    # BASE_IMAGE_NUM = 1084
    # 图像数据增强放大倍数，总数=BASE_IMAGE_NUM*GEN_NUM
    # GEN_NUM = 400
    # 训练集，验证集，测试集比例
    # PERCENT = {'TRAIN': 0.99, 'VALI': 0.01}


def show_densenetFlag_info():
    print('稠密神经网络参数如下：')
    if (os.path.exists(DFLAG.MODEL_PATH)):
        print('模型路径为：' + DFLAG.MODEL_PATH)
    else:
        print('模型位置不存在，请检查DensenetFlag.py设置是否正确。' + DFLAG.MODEL_PATH)
    print('可识别字符种类为：' + str(DFLAG.NCLASS) + '\n'
                                            '是否使用BLSTM：' + str(DFLAG.IS_USE_BLSTM) + '\n'
                                                                                     '是否使用GPU：' + str(
        DFLAG.IS_USE_GPU) + '\n'
          )


if __name__ == '__main__':
    show_densenetFlag_info()
