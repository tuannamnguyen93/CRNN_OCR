import pickle
import os
import numpy as np
import cv2

dict = {}
for i in range(0,len(os.listdir("../data/Seq_data/"))):
    if i % 100 == 0:
        for j in range(i-100, i):
            img = cv2.imread("../data/Seq_data/" + os.listdir("../data/Seq_data/")[j])
            img = img / 255.
            img = cv2.resize(img, (156, 32))
            # print os.listdir("../data/Seq_data/")[j]
            dict.update({ os.listdir("../data/Seq_data/")[j] : img})
        with open("../data/sequences_imgs.txt", "a") as f:
            f.write(pickle.dumps(dict))
    print i

# with open("../data/sequences_imgs.txt","w") as f:
#     f.write(pickle.dumps(dict))