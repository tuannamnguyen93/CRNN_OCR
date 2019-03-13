import pickle
from ast import literal_eval
import os
import numpy as np
import time

# with open("../data/sequences.txt", "r") as f:
#     seq_labels = pickle.load(f)
# with open("../data/sequences_features.txt", "r") as f2:
#     seq_features = pickle.load(f2)
#
# def get_data(a,b):
#     list1 = []
#     list2 = []
#     for i in range(a, b):
#         list1.append(seq_features["Sequence_" + str(i) + ".png"])
#         list2.append(seq_labels["Sequence_" + str(i) + ".png"])
#
#     arr1 = np.asarray(list1, dtype=np.float32)
#     arr2 = np.asarray(list2, dtype=np.float32)
#
#     return arr1, arr2

# new_dict = {}
# with open("../data/sequences.txt","r") as f:
#     seq = pickle.load(f)
#
# print seq
# with open("testfile.txt",'r') as f:
#     dict = f.read()
#     dict = literal_eval(dict)import pickle
# from ast import literal_eval
# import os
#
# def get_data(a,b):
#     with open("../data/sequences.txt","r") as f:
#         seq = pickle.load(f)
#         print seq
#
# get_data(None, None)


# #
# # print seq
# # with open("testfile.txt",'r') as f:
# #     dict = f.read()
# #     dict = literal_eval(dict)
# #
# # for keys, values in seq.items():
# #     list = []
# #     for i in values:
# #         for k, v in dict.items():
# #             if i == k:
# #                 list.append(v)
# #     new_dict.update({ keys:  list})
# #
# # with open("../data/sequences.txt","w") as f:
# #     f.write(pickle.dumps(new_dict))

def convert_dict():
    new_dict = {}
    with open("../data/sequences.txt","r") as f1:
        seq = pickle.load(f1)
    with open("testfile.txt",'r') as f2:
        dict = f2.read()
        dict = literal_eval(dict)

    for keys, values in seq.items():
        list = []
        for i in values:
            for k, v in dict.items():
                if i == k:
                    list.append(v)
        new_dict.update({ keys:  list})

    with open("../data/sequences.txt","w") as f:
        f.write(pickle.dumps(new_dict))

# new_dict = {}
#
# with open("testfile.txt",'r') as f:
#     dict = f.read()
#     dict = literal_eval(dict)
#
# for keys, values in dict.items():
#     keys = keys.split("_")[1]
#     new_dict.update({ keys : values })
#
# with open("testfile.txt",'w') as f:
#     f.write(str(new_dict))