from data_load import MultiFoodDataset,FoodDataset
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from model import Classifier
from utility import test_tfm, spec_transformers
import torchvision.transforms as transforms

batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
_dataset_dir = "./food11"
_exp_name = "sample"

print(device)
transformers = [spec_transformers([]),
                spec_transformers([transforms.RandomRotation(180)])]
weight = [0.6, 0.4]
# test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
test_set = MultiFoodDataset(os.path.join(_dataset_dir, "test"), tfms=transformers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = Classifier().to(device)
# 这里要加 , map_location='cpu'  因为GPU训练的模型转CPU会不兼容  这里用的参数 为kaggle 前几个版本的初始sample
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt", map_location='cpu'))
model_best.eval()
prediction = []

weight = np.array([0.6, 0.4])

with torch.no_grad():
    for data, _ in test_loader:
        # shape 为batch_size * num_of_classes
        test_pred = model_best(data.to(device))
        batch_len = test_pred.shape[0]
        print('test_pred', test_pred.shape)
        # shape 为batch_size
        oh = test_pred.cpu().data.numpy().reshape(batch_len//2, 2, 11)
        oh = np.dot(weight, oh)
        test_label = np.argmax(oh, axis=1)
        print(type(oh), oh.shape, oh)
        print('test_label', test_label.shape, test_label)
        prediction += test_label.squeeze().tolist()

# 初次预测结果
# [ 2  9  0  2  0  9  4  9  0  9  9  0  3  9  4  3  9 10  0  4  9 10  0  5
#   9  2 10 10  3  4  2  5  9  5 10  3 10  9  2  0  8  3  5  5  0  0  6  4
#   5  5  2  4  9 10  0  0  9  5  3  3  2  0  4  9]

print(prediction)


# create test csv
def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)


df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set)//2 + 1)]
df["Category"] = prediction
df.to_csv("submission.csv", index=False)
