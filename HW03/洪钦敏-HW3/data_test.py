from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    # transforms.ToTensor(),
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
        *[x for x in methods],
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.Resize((128, 128)),
        # transforms.ToTensor(),
    ])
    return tfm


if __name__ == '__main__':
    # 看看图片变幻效果
    pp = '0_106.jpg'
    im = Image.open(pp)
    my_trans = [spec_transformers([transforms.RandAugment()]),
                spec_transformers([transforms.RandomRotation(180),
                                transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                                transforms.RandomAutocontrast(),
                                transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                                                       saturation=(0.7, 1.3)),
                                transforms.RandomPerspective(distortion_scale=np.random.rand())]),
                spec_transformers([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                                transforms.RandomAdjustSharpness(2*np.random.rand()),
                                transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                                                       saturation=(0.7, 1.3))]),
                spec_transformers([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomAffine(degrees=30,translate=(0.1,0.2),shear=(0,30)),
                                transforms.RandomAdjustSharpness(2*np.random.rand()),
                                transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                                                       saturation=(0.7, 1.3))])
                ]
    axs = plt.figure().subplots(1, 5)
    axs[0].imshow(im)
    axs[0].set_title('src')
    axs[0].axis('off')

    trans_ims = [tr(im) for tr in my_trans]
    axs[1].imshow(trans_ims[0])
    axs[1].set_title('RandAugment')
    axs[1].axis('off')

    axs[2].imshow(trans_ims[1])
    axs[2].set_title('RandomInvert')
    axs[2].axis('off')

    axs[3].imshow(trans_ims[2])
    axs[3].set_title('RandomInvert')
    axs[3].axis('off')

    axs[4].imshow(trans_ims[3])
    axs[4].set_title('RandomInvert')
    axs[4].axis('off')

    plt.show()
