import os
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
# 本文件的主要目标在于了解声音数据
if __name__ == '__main__':
    split= 'train'
    phone_path = 'libriphone/feat'
    mode = 'train'
    label_dict = {}
    phone_file = open(r'E:\kpython\ML2022-Spring\HW02\洪钦敏-HW2\libriphone\train_split.txt').readlines()
    for line in phone_file:
        line = line.strip('\n').split(' ')
        label_dict[line[0]] = [int(p) for p in line[1:]]
    print(label_dict)

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(r'E:\kpython\ML2022-Spring\HW02\洪钦敏-HW2\libriphone\train_split.txt').readlines()
        random.seed(0)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * 0.8)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    print(usage_list)
    xxx = '2007-149877-0023 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 29 29 29 29 29 29 39 39 39 39 39 39 39 39 39 39 8 8 8 8 8 8 8 8 12 12 12 12 12 25 25 25 25 25 29 29 29 31 31 31 31 6 6 6 6 6 6 6 6 6 6 10 10 10 10 10 10 10 10 5 5 5 5 5 19 19 19 31 31 31 31 24 24 24 24 24 24 28 28 28 28 28 28 28 28 28 40 40 40 40 40 40 40 40 40 40 40 40 40 5 5 5 5 5 5 5 5 5 5 12 12 12 12 12 27 27 27 27 38 38 38 38 38 38 35 35 35 35 35 35 35 3 3 3 3 3 3 3 3 3 3 3 3 3 28 28 28 28 28 28 28 28 35 35 35 35 35 35 35 35 35 35 35 39 39 39 39 39 39 12 12 12 12 12 12 22 22 22 22 19 19 19 19 19 19 6 6 6 6 6 6 6 6 6 31 31 31 31 5 5 5 5 5 24 24 24 24 24 24 25 25 25 25 25 25 25 25 25 25 25 25 25 2 2 2 2 2 2 2 2 2 2 30 30 30 30 30 30 30 30 30 30 30 9 9 9 9 9 9 9 9 9 9 9 9 13 13 13 13 13 13 13 13 21 21 21 21 21 31 31 31 31 31 31 31 31 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 4 4 4 4 4 36 36 36 36 36 36 36 36 36 36 5 5 5 5 19 19 19 28 28 28 28 28 33 33 33 33 33 33 33 33 33 33 33 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 27 27 27 27 27 27 37 37 37 37 37 37 37 37 37 37 39 39 39 39 39 39 31 31 31 31 31 31 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 30 30 30 30 30 30 30 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 39 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    print(len(xxx.split(' '))-1) # 500
    feat = torch.load(r'E:\kpython\ML2022-Spring\HW02\洪钦敏-HW2\libriphone\feat\train\2007-149877-0023.pt')
    print(feat.shape) # 500 39
