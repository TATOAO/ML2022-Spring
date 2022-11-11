import torchvision.utils

_exp_name = "sample"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision.models import resnet18, vgg16

# This is for the progress bar.
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
import sys

myseed = 2333  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# img_grid = torchvision.utils.make_grid()

def imshow(imgs):
    # imgs = imgs / 2 + 0.5
    plt.imshow(imgs)
    plt.show()

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
# train_tfm = transforms.Compose([
#     # Resize the image into ap fixed shape (height = width = 128)
#     transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3./4,4./3)),
#     # 水平翻转图像
#     transforms.RandomHorizontalFlip(p=0.5),
#
#     transforms.Resize((128, 128)),
#     # You may add some transforms here.
#     # ToTensor() should be the last one of the transforms.
#     transforms.ToTensor(),
#     # normalize,
# ])

train_tfm = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    # 水平翻转图像
    transforms.RandomHorizontalFlip(p=0.5),
    # 旋转
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # normalize,
])


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# ResNet
class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)

        x1 = self.relu(x1)

        x2 = self.cnn_layer2(x1)

        x2 = self.relu(x2)

        x3 = self.cnn_layer3(x2)

        x3 = self.relu(x3)

        x4 = self.cnn_layer4(x3)

        x4 = self.relu(x4)

        x5 = self.cnn_layer5(x4)

        x5 = self.relu(x5)

        x6 = self.cnn_layer6(x5)

        x6 = self.relu(x6)

        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x6.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout



batch_size = 128
_dataset_dir = "./food11"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of training epochs and patience.
n_epochs = 50
patience = 300  # If no improvement in 'patience' epochs, early stop
lr = 0.0003
weight_decay = 1e-5
# Initialize a model, and put it on the device specified.
# model = Classifier().to(device)

# model = Residual_Network().to(device)
model = vgg16().to(device)
model2 = resnet18().to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

criterion2 = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=weight_decay)

# 设置模型的权重
weight_1 = 0.4
weight_2 = 0.6

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0


# 超参数
print('-- 训练参数 --')
print('seed %d' % myseed)
print('batch_size %d' % batch_size)
print('n_epochs %d' % n_epochs)
print('patience %d' % patience)
print('lr %f' % lr)
print('weight_decay %f' % weight_decay)
print('-------------')
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()
    model2.train()
    # These are used to record information in training.
    train_loss = []
    train_accs = []
    train_loss_2 = []
    train_accs_2 = []
    # 模型集成的准确率
    ensemble_accs = []
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)
        # for im in imgs:
        #     img_2 = functional.to_pil_image(im)
        #     plt.imshow(img_2)
        #     plt.show()

        # sys.exit(0)
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        logits2 = model2(imgs.to(device))
        ensemble_logits = logits * weight_1 + logits2 * weight_2
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))
        loss2 = criterion2(logits2, labels.to(device))
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        optimizer2.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        loss2.backward()
        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        grad_norm2 = nn.utils.clip_grad_norm_(model2.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        optimizer2.step()
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        acc2 = (logits2.argmax(dim=-1) == labels.to(device)).float().mean()
        ensemble_acc = (ensemble_logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        train_loss_2.append(loss2.item())
        train_accs_2.append(acc2)
        ensemble_accs.append(ensemble_acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    train_loss_2 = sum(train_loss_2) / len(train_loss_2)
    train_accs_2 = sum(train_accs_2) / len(train_accs_2)
    ensemble_accs = sum(ensemble_accs) / len(ensemble_accs)
    # Print the information.
    print(f"[ Train VGG16 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    print(f"[ Train Resnet18 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss_2:.5f}, acc = {train_accs_2:.5f}")
    print(f"[ Train Ensemble | {epoch + 1:03d}/{n_epochs:03d} ] loss = {ensemble_accs:.5f}, acc = {ensemble_accs:.5f}")
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    model2.eval()
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    valid_loss_2 = []
    valid_accs_2 = []
    # 模型集成的准确率
    ensemble_accs = []
    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))
            logits2 = model2(imgs.to(device))
            ensemble_logits = logits * weight_1 + logits2 * weight_2
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))
        loss2 = criterion2(logits2, labels.to(device))
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        acc2 = (logits2.argmax(dim=-1) == labels.to(device)).float().mean()
        ensemble_acc = (ensemble_logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        valid_loss_2.append(loss2.item())
        valid_accs_2.append(acc2)
        ensemble_accs.append(ensemble_acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    valid_loss_2 = sum(valid_loss_2) / len(valid_loss_2)
    valid_accs_2 = sum(valid_accs_2) / len(valid_accs_2)
    ensemble_accs = sum(ensemble_accs) / len(ensemble_accs)

    # Print the information.
    print(f"[ Valid VGG16 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    print(f"[ Valid Resnet18 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_2:.5f}, acc = {valid_accs_2:.5f}")
    print(f"[ Train Ensemble | {epoch + 1:03d}/{n_epochs:03d} ] loss = {ensemble_accs:.5f}, acc = {ensemble_accs:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

    # ensemble update logs
    if ensemble_accs > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Model_1 Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            print(f"[ Model_2 Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_2:.5f}, acc = {valid_accs_2:.5f} -> best")
            print(f"[ Ensemble Model_1 Valid | {epoch + 1:03d}/{n_epochs:03d} ] ensemble acc = {ensemble_accs:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            print(f"[ Model_2 Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss_2:.5f}, acc = {valid_accs_2:.5f} -> best")
    # ensemble save models
    if ensemble_accs > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        torch.save(model2.state_dict(), f"{_exp_name}_2_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = ensemble_accs
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# model_best = Classifier().to(device)
# model_best = Residual_Network().to(device)
model_best = vgg16().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()

model2_best = resnet18().to(device)
model2_best.load_state_dict(torch.load(f"{_exp_name}_2_best.ckpt"))
model2_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_pred_2 = model2_best(data.to(device))
        test_label = np.argmax((test_pred * weight_1 + test_pred_2 * weight_2).cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

# 超参数
print('-- 训练参数 --')
print('seed %d' % myseed)
print('batch_size %d' % batch_size)
print('n_epochs %d' % n_epochs)
print('patience %d' % patience)
print('lr %f' % lr)
print('weight_decay %f' % weight_decay)
print('-------------')