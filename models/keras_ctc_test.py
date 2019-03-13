import pickle
import numpy as np
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU,LSTM
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import itertools

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

def get_data(a,b):
    list1 = []
    list2 = []
    for i in range(a, b):
        list1.append(seq_features["Sequence_" + str(i) + ".png"])

        list2.append(seq_labels["Sequence_" + str(i) + ".png"])

    arr1 = np.asarray(list1, dtype=np.float32)
    arr2 = np.asarray(list2, dtype=np.int64)
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

model=load_model('first_try_LSTM_CTC.h5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
model3 = Model(inputs=model.input,outputs=model.get_layer('softmax').output)
print seq_labels['Sequence_90500.png']
X=seq_features['Sequence_90500.png']
X=np.expand_dims(X,axis=0)
print X.shape
s=model3.predict(X,batch_size=1)
