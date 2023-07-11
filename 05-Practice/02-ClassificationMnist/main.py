# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: main.py
@time: 2023/7/3 20:49
"""
# Import packages
import os
import hues
import time
import torch
from prettytable import PrettyTable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss

# Define hyperparameters
IMG_SIZE = 28  # 图像的总尺寸28*28
LABEL_NUM = 10  # 标签的种类数
BATCH_SIZE = 64  # 一个撮（批次）的大小，64张图片
DATA_PATH = './data/'

# Dataset
train_dataset = datasets.MNIST(
    root=DATA_PATH,
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = datasets.MNIST(
    root=DATA_PATH,
    train=False,
    transform=transforms.ToTensor()
)

# DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE
)


# Model object
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 隐藏层_0
        self.conv_0 = Sequential(  # 输入大小 (1, 28, 28)
            Conv2d(
                in_channels=1,  # 灰度图，只有一个通道
                out_channels=16,  # 要得到几多少个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为 (16, 28, 28)
            ReLU(),  # relu层
            MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )
        # 隐藏层_1
        self.conv_1 = Sequential(  # 下一个套餐的输入 (16, 14, 14)
            Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            ReLU(),  # relu层
            MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        # 隐藏层_2
        self.conv_2 = Sequential(
            Flatten(),
            Linear(32 * 7 * 7, 10)  # 全连接层得到的结果
        )
        # 模型网络结构
        self.nn = Sequential(
            self.conv_0,
            self.conv_1,
            self.conv_2
        )

    def forward(self, X):
        return self.nn(X)


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


def make_dir(dir_name=None):
    """
    Input a dir name, if the root dir don't have this dir, it will create this.
    :param dir_name: dir name
    :return: None
    """
    if dir_name and not os.path.exists(dir_name):
        os.mkdir(dir_name)


def get_percent(x: float) -> str:
    """
    Input a decimal in 0~1, return percentage format of this number.
    :param x: a decimal in 0~1
    :return: a*100(%)
    """
    return '{:.2f}%'.format(x * 100)


if __name__ == '__main__':
    # Get GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test model input and output
    print('+' * 40, 'Test model input and output'.center(20), '+' * 40)
    model = Model().to(device)
    demo_input = torch.ones(1, 1, 28, 28).to(device)
    hues.info(model)
    hues.log(model(demo_input).shape)
    print('+' * 100)

    # Loss_fun
    loss_fun = CrossEntropyLoss()
    loss_fun.to(device)

    # Optimizer
    Learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法
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
    EPOCHS = 20
    for i in range(EPOCHS):
        print('+' * 40, f'Epoch {i}'.center(20), '+' * 40)

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
    demo_input = torch.ones(1, 1, 28, 28).to(device)
    writer.add_graph(model, demo_input)
    writer.close()  # tensorboard --logdir='logs'
