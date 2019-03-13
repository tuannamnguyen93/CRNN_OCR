from tensorflow.python.keras.models import Model

import pickle
import numpy as np
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from tensorflow.python.keras.layers import Conv2D,Lambda,Flatten,Activation
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Lambda, Flatten, Activation, Input, Reshape, Dense, MaxPooling2D
from tensorflow.python.keras.layers import GRU, LSTM
from tensorflow.python.keras.layers import add, concatenate
from tensorflow.python.keras.optimizers import Adam, SGD


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


with open("sequences.txt", "r") as f:
    seq_labels = pickle.load(f)
with open("sequences_features.txt", "r") as f2:
    seq_features = pickle.load(f2)


def get_data(a, b):
    list1 = []
    list2 = []
    for i in range(a, b):
        list1.append(seq_features["Sequence_" + str(i) + ".png"])
        list2.append(seq_labels["Sequence_" + str(i) + ".png"])

    arr1 = np.asarray(list1, dtype=np.int64)
    arr2 = np.asarray(list2, dtype=np.float32)

    return arr1, arr2, np.ones((b - a, 1)) * 30, np.ones((b - a, 1)) * 5


input_data = Input(name='the_input', shape=(32, 128), dtype='float32')
lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(input_data)
lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1b')(input_data)
lstm_merge = add([lstm_1, lstm_1b])
lstm_2 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_merge)
lstm_2b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2b')(lstm_merge)
inner = Dense(73, kernel_initializer='he_normal', name='dense')(concatenate([lstm_2, lstm_2b]))
y_pred = Activation('softmax', name='softmax')(inner)
Model(inputs=input_data, outputs=y_pred).summary()
labels = Input(name='the_labels', shape=[5], dtype='int64')
input_length = Input(name='input_length', shape=[1], dtype='int64')
labels_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, labels_length])

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, labels_length], outputs=loss_out)
model.summary()
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
epochs = 10
batch_size = 32
# for e in range(0,epochs):
#   for s in range(0,int(80000/batch_size)):
input_data, labels, input_length, labels_length = get_data(1, 90001)
Y = np.zeros((90000, 1))
model.fit([input_data, labels, input_length, labels_length], Y, batch_size=32, epochs=1)
model.save('first_try_LSTM_CTC2.h5')

# model.summary()
