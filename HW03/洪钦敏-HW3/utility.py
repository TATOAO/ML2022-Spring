import os
import random
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Torchvision provides lots of useful utilities for image preprocessing
# data wrapping as well as data augmentation.
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

def spec_transformer(method):
    tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        method,
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        # transforms.ToTensor(),
    ])
    return tfm

def spec_transformers(methods):
    tfm = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        *[x for x in methods],
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
    ])
    return tfm
