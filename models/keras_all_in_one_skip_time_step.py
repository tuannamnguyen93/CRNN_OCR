import  itertools
import json
import tensorflow as tf
import numpy as np
import pickle
import joblib
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Masking, Conv2D, Lambda, Flatten, Activation, Input, Reshape, LSTM, Dense, add, concatenate,BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
import os
import cv2

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
#set_session(tf.Session(config=config))
with open("sequences_20_all_train_new_oc.txt", "r") as f:
    seq_labels = pickle.load(f)
with open("sequences_20_all_test_new_oc.txt", "r") as f:
    seq_labels_test = pickle.load(f)

#with open("../data/sequences_20_all_train_new.txt", "r") as f:
#    seq_labels_2 = pickle.load(f)
def skip_timestep(input):
    return  input[:,::5,:] 
def get_data(a,b):
    list1 = []
    list2 = []
    list3 = []
    for i in range(a, b):
        img = cv2.imread("../data/Seq_data_20_all_same_train_new/" + "Sequence_" + str(i) + ".png")
        #ratio = int(round((float(img.shape[1])/float(img.shape[0]))*32))
        img = cv2.resize(img,(581,32))
        list1.append(img)
	#seq=seq_labels["Sequence_" + str(i) + ".png"]
        #list2.append(seq)
        list2.append(seq_labels["Sequence_" + str(i) + ".png"])
	#list3.append(len(filter(lambda a: a != 72, seq)))
        print i

    arr1 = np.asarray(list1, dtype=np.float32)
    arr2 = np.asarray(list2, dtype=np.int64)

    return arr1, arr2,np.ones((b-a,1))*110,np.ones((b-a,1))*20
def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, :, :]
    

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = []
        for c in out_best:
            outstr.append(c)
        ret.append(outstr)
    return ret
#input_data, labels, input_length, labels_length = get_data(1,60001)

def model():
    input_data = Input(name='the_input_CNN', shape=(32,581,3), dtype='float32')
    
    inner = Conv2D(96, (9,9), padding='valid',name='conv1')(input_data)
    
    inner = BatchNormalization()(inner)
    
    inner=Lambda(max_out,arguments={'num_units':48},name='maxout1')(inner)
    
    inner = Conv2D(128, (9,9), padding='valid',name='conv2')(inner)
    
    inner = BatchNormalization()(inner)
    
    inner=Lambda(max_out,arguments={'num_units':64},name='maxout2')(inner)
    
    inner = Conv2D(256, (9,9), padding='valid',name='conv3')(inner)
    
    inner = BatchNormalization()(inner)
    
    inner=Lambda(max_out,arguments={'num_units':128},name='maxout3')(inner)
    
    inner = Conv2D(512, (8,8), padding='valid',name='conv4')(inner)
    
    inner = BatchNormalization()(inner)

    output=Lambda(max_out,arguments={'num_units':128},name='maxout4')(inner)
    
    print output.shape    
    

    
 #   inner = Conv2D(128, (8,8), padding='valid',name='conv5')(inner)

    output_maxout = Reshape((550,128))(output)
    
    output_maxout_resize=Lambda(skip_timestep,name='skip_timestep')(output_maxout)
    
#     inner2=Masking(mask_value=0., input_shape=(128, 128))(output_maxout)

    lstm_1=LSTM(256,return_sequences=True,kernel_initializer='he_normal',name='lstm1')(output_maxout_resize)

    lstm_1b=LSTM(256,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='lstm1b')(output_maxout_resize)

    lstm_add=add([lstm_1,lstm_1b])

    lstm_2=LSTM(256,return_sequences=True,kernel_initializer='he_normal',name='lstm2')(lstm_add)
    #
    lstm_2b=LSTM(256,return_sequences=True,go_backwards=True,kernel_initializer='he_normal',name='lstm2b')(lstm_add)

    lstm_merge=concatenate([lstm_2, lstm_2b])

    # inner=Dense(73,kernel_initializer='he_normal',name='dense')(concatenate([lstm_2,lstm_2b]))
    inner = Dense(3137, kernel_initializer='he_normal', name='dense')(lstm_merge)

    y_pred = Activation('softmax',name='softmax2')(inner)

    labels = Input(name='the_labels', shape=[20], dtype='int64')

    input_length = Input(name='input_length', shape=[1], dtype='int64')

    labels_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, labels_length])

    model = Model(inputs=[input_data, labels, input_length, labels_length], outputs=loss_out)
    
    
   

    return model
import sys
def train_model():
    m = model()


    
    m.load_weights("model9.hdf5",by_name=True)
    for layer in m.layers:
	layer.trainable = False
    for layer in m.layers[-13:]:
	layer.trainable = True 
    m.summary()
#    sys.exit(0)
    m.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=0.000001))
    img = cv2.imread("../data/Seq_data_20_all_same_test_new/" + "Sequence_" + "110" + ".png")
    img = cv2.resize(img,(581,32))
    img_S = np.expand_dims(img,axis=0)
#    print img_S.shape
    Y = np.zeros((50000, 1))
    for epoch in range(0,10):
        for i in range(0,16):
	  
            input_data, labels, input_length, labels_length = get_data(i*50000+1, (i+1)*50000+1)
            model2 = Model(inputs=m.get_layer('the_input_CNN').input,outputs=m.get_layer('softmax2').output)
	    S=model2.predict(img_S)
	    print  seq_labels_test["Sequence_110.png"]
            print decode_batch(S)
            print ("Epoch   "+str(epoch) +  "   :")
            m.fit([input_data, labels, input_length, labels_length], Y, batch_size=64)
        

            
	    m.save_weights("model9.hdf5")	
	m.save('first_try_CNN_LSTM_CTC_13.h5')
train_model()
