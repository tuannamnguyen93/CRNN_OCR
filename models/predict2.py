from keras.models import  Model
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
from keras.layers import Conv2D,Lambda,Flatten,Activation
from keras.models import load_model
from keras import Sequential
from keras.layers import Conv2D,Lambda,Flatten,Activation, Input
from keras.optimizers import Adam
import os
import pickle
test_datagen = ImageDataGenerator(rescale=1. / 255)
model=load_model('first_try_maxout.h5',custom_objects={'tf':tf})
model2 = Model(inputs=model.input,outputs=model.get_layer('maxout4').output)

def sliding_windows(img_path):
    X = cv2.imread(img_path)
    X=X/255.
    S=cv2.resize(X,(152,32))

    arr=[]
    for i in range(32):
        s=4*i
        arr.append(S[:,s:s+32,:])
    a = np.array(arr, dtype=np.float32)
    out=model2.predict(a,batch_size=32,verbose=1)
    print type(out)
    return np.reshape(out,(32,128))