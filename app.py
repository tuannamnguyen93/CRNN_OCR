from flask import Flask, render_template, request
import json
from keras.layers import Masking, Conv2D, Lambda, Flatten, Activation, Input, Reshape, LSTM, Dense, add, concatenate,BatchNormalization
import cv2
import numpy as np
import sys
sys.path.append(r'/home/sysadmin/HoangPM/JAP-OCR-NN/models/')
import keras.backend as K
import tensorflow as tf
from keras.models import Model,load_model
import pickle
from PIL import Image
import itertools
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def skip_timestep(input):
    return  input[:,::5,:]

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

def model():
    input_data = Input(name='the_input_CNN', shape=(32, 581, 3), dtype='float32')

    inner = Conv2D(96, (9, 9), padding='valid', name='conv1')(input_data)

    inner = BatchNormalization()(inner)

    inner = Lambda(max_out, arguments={'num_units': 48}, name='maxout1')(inner)

    inner = Conv2D(128, (9, 9), padding='valid', name='conv2')(inner)

    inner = BatchNormalization()(inner)

    inner = Lambda(max_out, arguments={'num_units': 64}, name='maxout2')(inner)

    inner = Conv2D(256, (9, 9), padding='valid', name='conv3')(inner)

    inner = BatchNormalization()(inner)

    inner = Lambda(max_out, arguments={'num_units': 128}, name='maxout3')(inner)

    inner = Conv2D(512, (8, 8), padding='valid', name='conv4')(inner)

    inner = BatchNormalization()(inner)

    output = Lambda(max_out, arguments={'num_units': 128}, name='maxout4')(inner)

    #   inner = Conv2D(128, (8,8), padding='valid',name='conv5')(inner)

    output_maxout = Reshape((550, 128))(output)

    output_maxout_resize = Lambda(skip_timestep, name='skip_timestep')(output_maxout)

    #     inner2=Masking(mask_value=0., input_shape=(128, 128))(output_maxout)

    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(output_maxout_resize)

    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1b')(
        output_maxout_resize)

    lstm_add = add([lstm_1, lstm_1b])

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_add)
    #
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2b')(lstm_add)

    lstm_merge = concatenate([lstm_2, lstm_2b])

    # inner=Dense(73,kernel_initializer='he_normal',name='dense')(concatenate([lstm_2,lstm_2b]))
    inner = Dense(3137, kernel_initializer='he_normal', name='dense')(lstm_merge)

    y_pred = Activation('softmax', name='softmax2')(inner)


    model = Model(inputs=input_data, outputs=y_pred)

    return model

app = Flask(__name__, template_folder='template')

# , static_url_path = "", static_folder = "static"
K.clear_session()
# model=model()
# model.load_weights("./models/model7.hdf5",by_name=True)

model=load_model("ditconme.h5",custom_objects={"tf":tf})

graph = tf.get_default_graph()

def get_seq(img):
    with open("models/sequences_20_all_same_test.txt", "r") as f:
        seq_labels = pickle.load(f)

    # model.summary()

    # img = cv2.imread("../test/20.png", 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh_img = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    # 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    dilated = cv2.dilate(thresh_img, kernel, iterations=3)

    ratio = int((float(dilated.shape[1])/dilated.shape[0])*32)
    img_arr = cv2.resize(dilated,(ratio,32))
    old_img = Image.fromarray(img_arr)

    new_img = Image.new('P', (581, 32))
    new_img.paste(old_img, (0, 0))

    new_img_arr = np.array(new_img)

    new_img_arr = cv2.cvtColor(new_img_arr,cv2.COLOR_GRAY2RGB)
    cv2.imwrite("a.png",new_img_arr)
    X = np.expand_dims(new_img_arr,axis=0)
    with graph.as_default():
        S = model.predict(X)

    with open("models/testfile3.txt", "r") as f:
        data = pickle.load(f)

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
    return phrase

test_folder = '../JAP-OCR-NN/test_app'
list_folder = list()
files = os.listdir(test_folder)
for i in range(1, len(files)):
    list_folder.append(files[i])

@app.route('/', methods=['GET'])
def index():
    return render_template("api.html",list=list_folder)

@app.route('/upload', methods=['POST','GET'])
def get_image():
    img_name = "imgtest.png"
    if request.method == 'POST':
        # K.clear_session()

        if not os.path.exists("../JAP-OCR-NN/static/"):
            os.mkdir("../JAP-OCR-NN/static/")

        if request.form.get("img_selected", False) != "":
            img_sel = request.form.get("img_selected", False)
            path_img = os.path.join(test_folder, img_sel)

            read_img = cv2.imread(path_img)
            read_img = 255 - read_img
            result = Image.fromarray(read_img.astype(np.uint8))
            phrase = get_seq(read_img)
        else:
            filestr = request.files['image'].read()
            npimg = np.fromstring(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            result = Image.fromarray(img.astype(np.uint8))

            phrase = get_seq(img)

        result.save("../JAP-OCR-NN/static/" + img_name, 'PNG')

    return render_template("upload.html", phrase = phrase)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=False)
