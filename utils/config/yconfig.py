import os

kerasTextModel = os.path.abspath(
    '..') + '\\text\\yoyo\\model\\' + 'text.h5'  ##keras版本模型权重文件
IMGSIZE = (608, 608)  ## yolo3 输入图像尺寸
## keras 版本anchors
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none', 'text', ]
## GPU选择及启动GPU序号
GPU = True  ##OCR 是否启用GPU
GPUID = 0  ##调用GPU序号
## nms选择,支持cython,gpu,python
nmsFlag = 'gpu'  ## cython/gpu/python ##容错性 优先启动GPU，其次是cpython 最后是python
if not GPU:
    nmsFlag = 'cython'
