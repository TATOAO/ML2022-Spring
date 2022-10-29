from torch.utils.data import Dataset
from utility import test_tfm
import os
from PIL import Image
from utility import spec_transformer
import torchvision.transforms as transforms


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label



# 重写数据载入  tfms 为transformers的集合
class MultiFoodDataset(Dataset):

    def __init__(self, path, tfms, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transforms = tfms
        # 图片重复次数
        self.pic_duplicate_times = len(tfms)

    def __len__(self):
        return self.pic_duplicate_times * len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx // self.pic_duplicate_times]
        im = Image.open(fname)
        im = self.transforms[idx % self.pic_duplicate_times](im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label
