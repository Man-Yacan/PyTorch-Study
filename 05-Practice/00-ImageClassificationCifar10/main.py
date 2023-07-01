# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: main.py
@time: 2023/6/27 16:47
"""

import os
import time
import hues
import glob
import torch

from prettytable import PrettyTable
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d, Linear, Flatten, functional, CrossEntropyLoss

LABEL_NAME = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 训练数据集增强
train_transformer = transforms.Compose([
    transforms.RandomCrop(28),  # 改变大小
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor()
])

test_transformer = transforms.Compose([
    transforms.Resize((28, 28)),  # 改变大小
    transforms.ToTensor()
])


# 定义读取数据集类
class MyDataset(Dataset):
    def __init__(self, img_list, transform=None, loader=None):
        super(MyDataset, self).__init__()

        self.transform = transform

        self.loader = loader
        if not self.loader:
            self.loader = self.image_loader

        self.img_info = []
        for path in img_list:
            img_label = path.split('\\')[-2]
            self.img_info.append((path, LABEL_NAME.index(img_label)))
            # img_info.append((path, label_dict[img_label]))

    def __getitem__(self, index):
        img_path, img_label = self.img_info[index]

        img_data = self.loader(img_path)
        if self.transform:
            img_data = self.transform(img_data)

        return img_data, img_label

    def __len__(self):
        return len(self.img_info)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')


# 获取全部的训练集图片路径以及测试集图片路径
img_train_list = glob.glob('./data/cifar-10-python/cifar-10-batches-py/train/*/*.png')
img_test_list = glob.glob('./data/cifar-10-python/cifar-10-batches-py/test/*/*.png')

train_dataset = MyDataset(img_train_list, transform=train_transformer)  # 测试集图片需要进行图片增强处理
test_dataset = MyDataset(img_test_list, transform=transforms.ToTensor())  # 训练集图片不需要

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    num_workers=2
)


class VGGNet(torch.nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.model = Sequential(
            # 00
            Conv2d(3, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),  # num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # 01
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # 02
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=1),
            # 03
            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            Linear(512 * 4, 10)
        )

    def forward(self, X):
        outputs = self.model(X)
        return outputs


def make_dir(dir_name=None):
    if dir_name and not os.path.exists(dir_name):
        os.mkdir(dir_name)


def train(dataloader, model, loss_fn, optimizer):
    sample_num = len(dataloader.dataset)  # Dataset中的样本数
    batch_num = len(dataloader)  # dataloader中的批量数
    model.train()
    train_loss, correct_num = 0, 0
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss_value, cur_all_sample_num = loss.item(), i * len(X)
            hues.log(f"Loss: {loss_value:>7f}  [{cur_all_sample_num:>5d}/{sample_num:>5d}]")

        correct_num += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    # 计算本轮训练的平均loss和训练正确率
    train_loss /= batch_num
    correct_num /= sample_num

    return train_loss, correct_num


# Training function
def test(dataloader, model, loss_fn):
    sample_num = len(dataloader.dataset)
    batch_num = len(dataloader)
    model.eval()
    test_loss, correct_num = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct_num += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= batch_num
    correct_num /= sample_num
    return test_loss, correct_num


def get_percent(x):
    return '{:.2f}%'.format(x * 100)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGGNet().to(device)
    hues.info(model)

    # Loss function
    loss_fun = CrossEntropyLoss()
    loss_fun.to(device)

    # Optimizer
    learning_rate = 0.0001
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=.9,
        weight_decay=5e-4
    )

    # 学习率的衰减, 每经过3轮更新梯度，学习率衰减为原来的0.9
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=.9
    )

    # Using tensorboard to visualize network structure
    LOG_DIR = 'logs'
    make_dir(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)

    # Training model
    EPOCHS = 100
    for i in range(EPOCHS):
        print('+' * 30, f'Epoch {i}'.center(10), '+' * 30)

        train_loss, train_cor = train(train_loader, model, loss_fun, optimizer)

        # 更新学习率
        cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        scheduler.step()

        # hues.info(f'Epoch {i}: Train Loss is {train_loss}, Train Accuracy is', '{:.2f}%.'.format(train_cor * 100),
        #           f'Learning Rate is {cur_lr}.')

        test_loss, test_cor = test(test_loader, model, loss_fun)
        # hues.info(f'Epoch {i}: Test Loss is {test_loss}, Test Accuracy is', '{:.2f}%.'.format(test_cor * 100))

        tb = PrettyTable(["Dataset", "Loss", "Accuracy"])
        tb.add_row(["Train", train_loss, get_percent(train_cor)])
        tb.add_row(["Test", test_loss, get_percent(test_cor)])
        print(tb)
        hues.info(f'Learning Rate is {cur_lr}.')

        # Writing values
        writer.add_scalar('Loss/Train', train_loss, i)
        writer.add_scalar('Loss/Test', test_loss, i)
        writer.add_scalar('Accuracy/Train', train_cor, i)
        writer.add_scalar('Accuracy/Test', test_cor, i)
        writer.add_scalar('Learning Rate', cur_lr, i)

    hues.success("Done!")

    # Saving model
    timer = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    MODEL_DIR = 'model'
    make_dir(MODEL_DIR)
    torch.save(model, f'{MODEL_DIR}/model_{timer}.pth')

    # 神经网络模型的可视化
    demo_input = torch.ones(6, 3, 28, 28).to(device)  # batch_size=6, in_channels=3, height=28, weight=28.
    writer.add_graph(model, demo_input)
    writer.close()  # tensorboard --logdir='logs'
