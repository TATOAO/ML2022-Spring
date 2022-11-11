
# 


 nn.Conv2d(3, 64, 3, 1, 1)
 输入 通道数 3
 输出 通道数 64，
 卷积核 3 * 3
 步长 为 1
 padding 填充 1

```python

torch.nn.Conv2d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None)
```



# 1、第一次训练

未修改模型， 修改了图片增强的方法
```python
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

```

训练参数

```bash
-- 训练参数 --

seed 6666
batch_size 64
n_epochs 50
patience 300
lr 0.000300
weight_decay 0.000010
-------------

```

# 2、第二次训练

增大 batch_size, 延长训练次数


```python
-- 训练参数 --
seed 6666
batch_size 128
n_epochs 100
patience 300
lr 0.000300
weight_decay 0.000010
-------------

```

kaggle 评分

```
Score: 0.76195
Private score: 0.73282
```

# 3、第三次训练

修改模型,使用示例中的 Residual_Network ,降低 batch_size


```python
-- 训练参数 --
seed 6666
batch_size 64
n_epochs 100
patience 300
lr 0.000300
weight_decay 0.000010
```
acc = 0.69709 -> best

# 4、第四次训练

修改模型，使用 `resnet18`

```python
-- 训练参数 --
seed 6666
batch_size 64
n_epochs 100
patience 300
lr 0.000300
weight_decay 0.000010

```
0.72151


# 5、第五次训练
修改模型，使用 `resnet18`，延长训练时间 200个 epoch

```python
-- 训练参数 --
seed 6666
batch_size 64
n_epochs 200
patience 300
lr 0.000300
weight_decay 0.000010
```

测试集准确率：`0.75332`
训练集准确率：`0.90812`

kaggle 结果

```python
private_score: 0.77934
public_score: 0.77988

```


# 6、第六次训练
使用模型集成
