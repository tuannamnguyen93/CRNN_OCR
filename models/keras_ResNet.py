import json
import tensorflow as tf
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Conv2D, Lambda, Activation, Input, Reshape, LSTM, Dense, add, concatenate, MaxPooling2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras import backend as K
import os
import cv2
import sys

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

with open("sequences_20_all_same_train.txt", "r") as f:
    seq_labels = pickle.load(f)

def get_data(a, b):
    list1 = []
    list2 = []
    ratio = 0
    for i in range(a, b):
        img = cv2.imread("../data/Seq_data_20_all_same_train/" + "Sequence_" + str(i) + ".png")
        ratio = int(round((float(img.shape[1]) / float(img.shape[0])) * 32))
        img = cv2.resize(img, (ratio, 32))
        list1.append(img)
        list2.append(seq_labels["Sequence_" + str(i) + ".png"])
        print i

    arr1 = np.asarray(list1, dtype=np.float32)
    arr2 = np.asarray(list2, dtype=np.int64)

    return arr1, arr2, np.ones((b - a, 1)) * 572, np.ones((b - a, 1)) * 20, ratio

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, 2:, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#ResNet34
def residual_block(layer, n_c, s=(1, 1)):
    shortcut = layer

    inner = Conv2D(n_c, (3, 3), strides=s, padding='same')(layer)

    inner = BatchNormalization()(inner)

    inner = LeakyReLU()(inner)

    inner = Conv2D(n_c, (3, 3), strides=s, padding='same')(inner)

    inner = BatchNormalization()(inner)

    if s != (1,1):
        shortcut = Conv2D(n_c, (1, 1), strides=s, padding='same')(shortcut)

        shortcut = BatchNormalization()(shortcut)

    output = add([shortcut,inner])

    relu_block = LeakyReLU()(output)

    return relu_block

def model():
    _, _, _, _, ratio = get_data(1, 2)

    input = Input(name='the_input', shape=(32, ratio, 3), dtype='float32')

    inner = Conv2D(16, (7, 7), padding='same', strides=(2, 2))(input)

    inner = BatchNormalization()(inner)

    inner = LeakyReLU()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 2))(inner)

    for i in range(3):
        s = (2, 2) if i == 0 else (1, 1)
        inner = residual_block(inner, 16, s=s)

    for i in range(4):
        s=(2,2) if i==0 else (1,1)
        inner = residual_block(inner, 32, s=s)

    for i in range(6):
        s = (2, 2) if i == 0 else (1, 1)
        inner = residual_block(inner, 64, s=s)

    for i in range(3):
        s = (2, 2) if i == 0 else (1, 1)
        inner = residual_block(inner, 128, s=s)

    inner = GlobalAveragePooling2D()(inner)
    print inner.shape
    sys.exit(0)
    output = Reshape([576, 128])(inner)

    lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(output)

    lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1b')(output)

    lstm_merge = concatenate([lstm_1, lstm_1b])

    inner = Dense(3137, kernel_initializer='he_normal', name='dense')(lstm_merge)

    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[20], dtype='int64')

    input_length = Input(name='input_length', shape=[1], dtype='int64')

    labels_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, labels_length])

    model = Model(inputs=[input, labels, input_length, labels_length], outputs=loss_out)

    model.summary()

    return model

def train_model():
    m = model()

    # optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    m.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    Y = np.zeros((50000, 1))
    for epoch in range(0, 2):
        print epoch
        for i in range(0, 20):
            input_data, labels, input_length, labels_length, _ = get_data(i * 50000 + 1, (i + 1) * 50000 + 1)

            m.fit([input_data, labels, input_length, labels_length], Y, batch_size=32, epochs=1)

            m.save('first_try_CNN_LSTM_CTC_12.h5')

train_model()
