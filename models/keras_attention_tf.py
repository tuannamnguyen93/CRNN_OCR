import tensorflow as tf
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Conv2D, Lambda, Flatten, Activation, Input, Reshape, LSTM, Dense, add, concatenate, MaxPooling2D, BatchNormalization, RepeatVector, Permute, merge
from keras.optimizers import Adam, SGD
from keras import backend as K
import cv2
import sys
import os
from attention_decoder import AttentionDecoder
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# with open("../data/sequences_20_all_train_new.txt", "r") as f:
#     seq_labels = pickle.load(f)

def get_data(a, b):
    list1 = []
    list2 = []
    for i in range(a, b):
        try:
            img1 = cv2.imread("../data/Seq_data_20_all_same_train_new/" + "Sequence_" + str(i) + ".png")
            img2 = cv2.imread("../data/Seq_data_20_all_same_test_new/" + "Sequence_" + str(i) + ".png")
            img1 = cv2.resize(img1, (581, 32))
            img2 = cv2.resize(img2, (581, 32))
        except:
            pass
        list1.append(img1)
        list2.append(img2)
        # list2.append(seq_labels["Sequence_" + str(i) + ".png"])
        print i

    arr1 = np.asarray(list1, dtype=np.float32)
    arr2 = np.asarray(list2, dtype=np.float32)
    # arr2 = np.asarray(list2, dtype=np.int64)

    return arr1, arr2

def model():
    input = Input(name='the_input', shape=(32, 581, 3), dtype='float32')

    inner = Conv2D(16, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv1')(input)

    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), name='pool1')(inner)

    inner = Conv2D(32, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv2')(inner)

    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), name='pool2')(inner)

    inner = Conv2D(64, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv3')(inner)

    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), name='pool3')(inner)

    inner = Conv2D(64, (3, 3), padding='same', activation="relu", data_format="channels_last", name='conv4')(inner)

    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), name='pool4')(inner)

    inner = Conv2D(128, (2, 2), padding='same', activation='relu', data_format="channels_last", name='conv5')(inner)

    inner = BatchNormalization()(inner)

    inner = MaxPooling2D(pool_size=(2, 1), name='pool5')(inner)

    output = Reshape([581, 128])(inner)

    lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(output)

    lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1b')(output)

    lstm_add = add([lstm_1, lstm_1b])

    lstm_2 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_add)

    lstm_2b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2b')(lstm_add)

    lstm_merge = concatenate([lstm_2, lstm_2b])

    # # attention = Permute([2, 1])(lstm_merge)
    #
    # attention = Dense(1,activation="tanh")(lstm_merge)
    #
    # attention = Flatten()(attention)
    #
    # attention = Activation("softmax")(attention)
    #
    # attention = RepeatVector(256)(attention)
    #
    # attention = Permute([2,1])(attention)
    #
    # activations = merge([lstm_merge,attention],mode="mul")
    #
    # # attention_context = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(256,))(activations)
    #
    # a = LSTM(256, return_sequences=False)(activations)
    #
    # prob = Dense(3136)(a)
    #
    # att = Activation('softmax')(prob)

    att = AttentionDecoder(256,3136)(lstm_merge)

    a = LSTM(3136, return_sequences=False)(att)

    prob = Dense(3136)(a)

    y_pred = Activation('softmax')(prob)

    model = Model(inputs=input, outputs=y_pred)

    model.summary()

    return model

def train_model():
    m = model()

    # m.load_weights("checkpoint_CNN/weights-05-0.20.hdf5", by_name=True)
    #
    # for layer in m.layers:
    #     layer.trainable = False
    # for layer in m.layers[-16:]:
    #     layer.trainable = True
    # optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    m.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # Y = np.zeros((50000, 1))
    for epoch in range(0, 5):
        print epoch
        for i in range(0, 100):
            # input_data, labels, _ = get_data(i * 50000 + 1, (i + 1) * 50000 + 1)
            x, y, = get_data(i * 10000 + 1, (i + 1) * 10000 + 1)

            # m.fit([input_data, labels], Y, batch_size=64)
            m.fit(x, y, batch_size=64)

            m.save_weights("model_attention.hdf5")

        m.save('first_try_attention.h5')

train_model()
