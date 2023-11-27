import os
import json, copy
import argparse
import shutil

annnotation = "annotation_2.json"

dir_path = []
cur_path = os.path.dirname(__file__)

annotation_file = os.path.join(cur_path,annnotation)
annotation_path = os.path.join(cur_path,"annotation")
f = open(annotation_file)

data_f = json.load(f)

for i in data_f:
    file_exist = os.path.join(annotation_path,i)
    if not os.path.exists(file_exist):
        os.makedirs(file_exist)
    for x in data_f[i]:
        print(x,i)
        shutil.copy(x,file_exist)
    print("done\n")
f.close()

