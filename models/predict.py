from keras.models import Model, load_model
import pickle
import numpy as np
import itertools
from keras import backend as K
import tensorflow as tf
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, 2:, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

with open("sequences_20_same_test.txt", "r") as f:
    seq_labels = pickle.load(f)

model = load_model('first_try_CNN_LSTM_CTC_3.h5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred,'tf':tf})

model3 = Model(inputs=model.get_layer('the_input').input,outputs=model.get_layer('softmax').output)

model3.summary()

# print seq_labels['Sequence_90601.png']
import cv2
X=cv2.imread("../data/Seq_data_20_same_test/Sequence_10.png")
X=cv2.resize(X,(581,32))
# Y=arr2_r[70347]

# print Y
# cv2.imshow("fd",X)
# cv2.waitKey(0)

X = np.expand_dims(X,axis=0)
S = model3.predict(X)
print S.shape

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = []
        for c in out_best:
            outstr.append(c)
        ret.append(outstr)
    return ret

print seq_labels["Sequence_10.png"]
print decode_batch(S)

# S,labels,_,_=get_data(90000,100000)
#
# V=model3.predict(S,batch_size=10000)
#
# pred = decode_batch(V)
#
# error=0
# for i in range(10000):
#     t=0
#     for s in range (len(pred[i])):
#         if t==5:
#             break
#         if pred[i][s]==72:
#             continue
#         print labels[i][t]
#         print pred[i][s]
#         if pred[i][s]!=labels[i][t]:
#             error+=1
#         t+=1
# print error
