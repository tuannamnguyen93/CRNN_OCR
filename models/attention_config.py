import json
import pickle
import cv2
import numpy as np
import sys

dick = {}
VOCAB = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2,'<PAD>':3}
VOC_IND = {}

with open('testfile2.txt') as f:
    d = f.read()
    f.close()
d = json.loads(d)

for k, v in d.items():
    dick[v] = k
for i in range(len(dick)):
    VOCAB[dick[i]] = i + 4
for key, value in VOCAB.items():
    VOC_IND[value] = key
# sys.exit(0)

MAX_LEN_WORD = 30
VOCAB_SIZE = len(VOCAB)
BATCH_SIZE = 256
RNN_UNITS = 256
EPOCH = 10000
DISPLAY_STEPS = 10
LOGS_PATH = '../models/attention/log'
SAVE_PATH = '../models/attention/save_model'
train_dir="../data/Seq_data_20_all_same_train_new/"
val_dir="../data/Seq_data_20_all_same_test_new/"

def label2int(label):#label_shape (num,len)
    target_input=np.ones((len(label), MAX_LEN_WORD),dtype=np.int64) +2
    target_out = np.ones((len(label), MAX_LEN_WORD),dtype=np.int64) +2
    for i in range(len(label)):
        target_input[i][0]=0
        for j in range(len(label[i])):
            target_input[i][j+1]=label[i][j]+4
            target_out[i][j]=label[i][j]+4
        target_out[i][len(label[i])]=1

    print target_out

    return target_input,target_out

def int2label(decode_label):
    print decode_label
    label=[]
    for i in range(decode_label.shape[0]):
        temp=[]
        for j in range(decode_label.shape[1]):
            if VOC_IND[decode_label[i][j]]=='<EOS>':
                break
            elif decode_label[i][j]==3:
                continue
            else:
                temp.append(VOC_IND[decode_label[i][j]])
        label.append(temp)
    return label

def get_data(data_dir,labels,a,b):
    list1 = []
    list2 = []
    for i in range(a, b):
        img = cv2.imread(data_dir + "Sequence_" + str(i) + ".png",0)
        if img is not None:
            ratio = int(round((float(img.shape[1])/float(img.shape[0]))*32))
            img = cv2.resize(img,(ratio,32))
            img = img.swapaxes(0, 1)
            img = 255-img
            list1.append(np.array(img[:, :, np.newaxis]))
            list2.append(labels["Sequence_" + str(i) + ".png"])
            print i
        else:
            pass

    arr1 = np.array(list1)

    return arr1, list2

def cal_acc_char(pred,label):
    num=0
    count=0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            try:
                if pred[i][j]==label[i][j]:
                    num+=1
                    count+=1
                else:
                    count+=1
            except:
                pass
    if count == 0:
        return 0.0
    return num*1.0/count
