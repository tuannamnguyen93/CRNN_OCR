
from flask import Flask, render_template, request
import json
from keras.layers import Masking, Conv2D, Lambda, Flatten, Activation, Input, Reshape, LSTM, Dense, add, concatenate,BatchNormalization
import cv2
import numpy as np
import sys
sys.path.append(r'/home/sysadmin/HoangPM/JAP-OCR-NN/models/')
# import predict3
import keras.backend as K
import tensorflow as tf
from keras.models import Model,load_model
import pickle
from PIL import Image
import itertools
import os

with open("models/sequences_20_all_same_test.txt", "r") as f:
    seq_labels = pickle.load(f)
from flask import Flask, render_template, request
import json
from keras.layers import Masking, Conv2D, Lambda, Flatten, Activation, Input, Reshape, LSTM, Dense, add, concatenate,BatchNormalization
import cv2
import numpy as np
import sys
sys.path.append(r'/home/sysadmin/HoangPM/JAP-OCR-NN/models/')
# import predict3
import keras.backend as K
import tensorflow as tf
from keras.models import Model,load_model
import pickle
from PIL import Image
import itertools
import os

# model.summary()
model=load_model("ditconme.h5",custom_objects={"tf":tf})
img = cv2.imread("./test_app/Sequence_1007.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img= cv2.resize(img, (581, 32))
# ret, thresh_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
# #
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
# dilated = cv2.dilate(thresh_img, kernel, iterations=2)
#
# ratio = int((float(dilated.shape[1]) / dilated.shape[0]) * 32)
# img_arr = cv2.resize(dilated, (ratio, 32))
# old_img = Image.fromarray(img_arr)
#
# new_img = Image.new('P', (581, 32))
# new_img.paste(old_img, (0, 0))
#
# new_img_arr = np.array(new_img)
#
# new_img_arr = cv2.cvtColor(new_img_arr, cv2.COLOR_GRAY2RGB)
# cv2.imwrite("a.png", new_img_arr)
X = np.expand_dims(img, axis=0)

S = model.predict(X)

with open("models/testfile3.txt", "r") as f:
    data = pickle.load(f)

ret = []
for j in range(S.shape[0]):
    out_best = list(np.argmax(S[j, :], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = []
    for c in out_best:
        outstr.append(c)
    ret.append(outstr)
print ret
# sys.exit(0)
phrase = ""
for i in ret:
    for j in i:
        for k, v in data.items():
            if j == v:
                if len(str(k)) < 4:
                    e = bytearray.fromhex("%x" % int(k))
                    g = e.decode('shift_jis')
                else:
                    e = '\033$B' + bytearray.fromhex("%x" % int(k))
                    g = e.decode('iso2022_jp')
                phrase += g
        if j == 3136:
            t = " "
            phrase += t

print phrase