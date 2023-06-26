# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: main.py
@time: 2023/6/25 15:50
"""

import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss


# 使用Sequential来优化网络结构
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.model = torch.nn.Sequential(
            Conv2d(3, 32, 5, padding=2),  # Input: 3@32×32, Output: 32@32×32
            MaxPool2d(2),  # Input: 32@32×32, Output: 32@16×16
            Conv2d(32, 32, 5, padding=2),  # Input: 32@16×16, Output: 32@16×16
            MaxPool2d(2),  # Input: 32@16×16, Output: 32@8×8
            Conv2d(32, 64, 5, padding=2),  # Input: 32@8×8, Output: 64@8×8
            MaxPool2d(2),  # Input: 64@8×8, Output: 64@4×4
            Flatten(),  # Input: 64@4×4, Output: 1024
            Linear(1024, 64),  # Input: 1024, Output: 64
            Linear(64, 10)  # Input: 64, Output: 10
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

DATA_DIR = '../05-Transforms/data/'

test_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    transform=transformer,
    train=False,
    download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

if __name__ == '__main__':
    module = MyModule()
    loss = CrossEntropyLoss()

    for i, data in enumerate(test_loader):
        imgs, targets = data
        # print(targets.shape)
        outputs = module(imgs)
        # print(outputs.shape)
        res_loss = loss(outputs, targets)
        print(res_loss)  # 打印交叉熵
        res_loss.backward()
