from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Bidirectional, LSTM
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Activation, Permute
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import concatenate
from keras.regularizers import l2


# 定义卷积块

def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    """
    卷积块
    :param input: 上层输入feather maps
    :param growth_rate: 输出feather maps数量
    :param dropout_rate: 随机切断率
    :param weight_decay:
    :return:
    """
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    """
稠密块
    :param x: 上层输入feather maps
    :param nb_layers: 稠密块中的卷积层数目
    :param nb_filter: 总feather maps数目
    :param growth_rate:
    :param droput_rate:
    :param weight_decay:
    :return:
    """
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=2, weight_decay=1e-4):
    """
稠密连接层
    """
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


# 定义双向RNN
def dense_blstm(input):
    rnnunit = 256
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm1')(input)
    x = Dense(rnnunit, name='blstm1_out', activation='linear')(x)
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm2')(x)
    return x


# 识别数字默认为10+1种类
def densenet(input, nclass=11, use_blstm=False):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv _nb_filter=64 filter_size=5*5 strides=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # _nb_filter=64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # _nb_filter=128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # _nb_filter=128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # _nb_filter=192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # _nb_filter=128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    if (use_blstm):
        x = dense_blstm(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)  # 产生三维数据[batch_size,width,nclass]

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred
