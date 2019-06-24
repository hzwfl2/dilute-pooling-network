# model script
"""
@author: mengxue.Zhang
"""

from keras.models import Model
from keras.optimizers import Adam
from keras import losses
from keras.layers import Dropout, Activation, add, Input, Concatenate , Lambda,  Reshape
from keras.layers import Conv2D, MaxPooling2D


image_shape=[88, 88]
channel = 1
padding = 'valid'


def get_model(classes=10, lr=0.001):
    while True:

        model = DP_model(width=image_shape[0], height=image_shape[1], channel=channel, classes=classes)
        optimizer = Adam(lr=lr, decay=0.0)

        model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        yield model


def DP_model(width, height, channel, classes):
    inpt = Input(shape=(width, height, channel), name='0_input')

    x = Conv2D(16, kernel_size=(5, 5), padding=padding, activation='relu', name='1_conv')(inpt)
    x = Rectified_Pooling(inpt=x, filters=16, couple=False)

    x = Conv2D(32, kernel_size=(5, 5), padding=padding, activation='relu', name='2_conv')(x)
    x = Rectified_Pooling(inpt=x, filters=32, couple=False)

    x = Conv2D(64, kernel_size=(6, 6), padding=padding, activation='relu', name='3_conv')(x)
    x = Rectified_Pooling(inpt=x, filters=64, couple=True)

    x = Conv2D(128, kernel_size=(5, 5), padding=padding, activation='relu', name='4_conv')(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(classes, kernel_size=(3, 3), padding=padding, activation=None, name='5_conv')(x)
    x = Reshape([classes])(x)
    x = Activation('softmax')(x)

    models = Model(inputs=inpt, outputs=x)
    return models


def Rectified_Pooling(inpt, ps=2, filters=32, add_max=True, couple=False, dropout=True):
    input_size = inpt.get_shape()[1]

    xmax = MaxPooling2D(pool_size=(ps, ps), strides=(ps, ps), name=str(filters)+'_max')(inpt)

    xs = []
    for i in range(ps):
        for j in range(ps):
            x_ = Lambda(lambda x: x[:, i:input_size:ps, j:input_size:ps, :])(inpt)
            xs.append(x_)

    xs = Concatenate(axis=-1)(xs)
    if couple:
        xs = Dropout(rate=0.5)(xs)

    # linear combination
    xs = Conv2D(filters, kernel_size=(1, 1), padding=padding, activation=None, name=str(filters)+'_1conv1')(xs)
    if dropout:
        xs = Dropout(rate=0.5, name=str(filters)+'_delta')(xs)
    if add_max:
        xs = add(inputs=[xs, xmax], name=str(filters)+'_sum')
    else:
        xs = xs
    return xs
