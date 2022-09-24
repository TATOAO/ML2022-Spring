# Numerical Operations
import math
import time

import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# random_split随机将一个数据集分割成给定长度的不重叠的新数据集。可选择固定生成器以获得可复现的结果

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

from utility import *
from config import config
from dataset import COVID19Dataset
from model import trainer, My_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Choose features you deem useful by modifying the function below.
def select_feat(train_data, valid_data, test_data, select_all=True, feature_idx=None):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # feat_idx = list(range(raw_x_train.shape[1]))[1:] # TODO: Select suitable feature columns.
        feat_idx = feature_idx
        # print(feat_idx)

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


if __name__ == '__main__':
    # Set seed for reproducibility
    same_seed(config['seed'])

    features = pd.read_csv('./covid.train.csv')
    x_data, y_data = features.iloc[:, :-1], features.iloc[:, -1]
    k=config['feature_k']
    selector = SelectKBest(score_func=f_regression, k=config['feature_k'])
    result = selector.fit(x_data, y_data)
    idx = np.argsort(result.scores_)[::-1]
    print(x_data.columns[idx[:k]])
    selected_idx = list(np.sort(idx[:k]))
    print(selected_idx)
    time.sleep(3)

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'],
                                                             selected_idx)

    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    # init dataset
    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)

    model = My_Model(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_pred(preds, config['pred_path'])
