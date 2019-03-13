import json
from keras.models import Model,load_model
import pickle
import numpy as np
import itertools
from keras import backend as K
import tensorflow as tf
from keras.layers import Conv2D, Lambda, Activation, Input, Reshape, LSTM, Dense, add, concatenate,BatchNormalization
import cv2
from PIL import Image



def get_seq(img,model,graph):
    with open("models/sequences_20_all_same_test.txt", "r") as f:
        seq_labels = pickle.load(f)


    # model.summary()

    # img = cv2.imread("../test/20.png", 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh_img = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    ratio = int((float(dilated.shape[1])/dilated.shape[0])*32)
    img_arr = cv2.resize(dilated,(ratio,32))
    old_img = Image.fromarray(img_arr)

    new_img = Image.new('P', (581, 32))
    new_img.paste(old_img, (0, 0))

    new_img_arr = np.array(new_img)

    new_img_arr = cv2.cvtColor(new_img_arr,cv2.COLOR_GRAY2RGB)

    X = np.expand_dims(new_img_arr,axis=0)
    with graph.as_default():
        S = model.predict(X)


    with open("models/testfile2.txt",'r') as f:
        data = json.load(f)

    ret = []
    for j in range(S.shape[0]):
        out_best = list(np.argmax(S[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = []
        for c in out_best:
            outstr.append(c)
        ret.append(outstr)

    phrase = ""
    for i in ret:
        for j in i:
            for k, v in data.items():
                if j == v:
                    if len(k.split("_")[1]) < 4:
                        e = bytearray.fromhex("%x" % int(k.split("_")[1]))
                        g = e.decode('shift_jis')
                    else:
                        e = '\033$B' + bytearray.fromhex("%x" % int(k.split("_")[1]))
                        g = e.decode('iso2022_jp')
                        phrase += g
            if j == 3136:
                t = " "
                phrase += t

    print phrase
    return phrase


