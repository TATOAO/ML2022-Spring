from PIL import Image
import torchvision.transforms as transforms

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
        transforms.Resize((128, 128)),
        *[x for x in methods],
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        # transforms.ToTensor(),
    ])
    return tfm


if __name__ == '__main__':
    # 看看图片变幻效果
    pp = './food11/validation/8_248.jpg'
    im = Image.open(pp)
    im.show()
    x = train_tfm(im)
    # x = spec_transformer(transforms.RandomVerticalFlip(0))(im)
    x = spec_transformers([])(im)
    x.show()
    print(x.shape)
    y = test_tfm(im)
    print(y.shape)