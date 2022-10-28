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




class FoodDataset2(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return 4*len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx//4]
        im = Image.open(fname)
        if idx % 4 == 0:
            im = spec_transformer(transforms.RandomVerticalFlip(0))(im)
        elif idx % 4 == 1:
            im = spec_transformer(transforms.RandomRotation(180))(im)
        elif idx % 4 == 2:
            im = spec_transformer(transforms.RandomHorizontalFlip(1))(im)
        elif idx % 4 == 3:
            im = spec_transformer(transforms.RandomVerticalFlip(1))(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label
