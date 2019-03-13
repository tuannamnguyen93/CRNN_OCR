from keras.models import  Model
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import pickle
import sys

test_datagen = ImageDataGenerator(rescale=1. / 255)
model=load_model('first_try_maxout.h5',custom_objects={'tf':tf})
model2 = Model(inputs=model.input,outputs=model.get_layer('maxout4').output)

def sliding_windows(img_path):
    X = cv2.imread(img_path)
    # X=X/255.
    ratio = int(round((float(X.shape[1]) / float(X.shape[0])) * 32))
    S = cv2.resize(X,(ratio,32))
    arr=[]

    for i in range((ratio-32)/4 + 1):
        s = 4 * i
        if (s+32) > ratio:
            break
        arr.append(S[:,s:s+32,:])
    a = np.array(arr, dtype=np.float32)
    out=model2.predict(a,batch_size=32,verbose=1)
    return np.reshape(out,(32,128))

dict = {}
t=0
for i in os.listdir("../data/Seq_data_20_train/"):
    dict.update({ i : sliding_windows("../data/Seq_data_20_train/"+i)})
    print t
    t+=1

with open("../data/sequences_20_features.txt","w") as f:
    f.write(pickle.dumps(dict))
