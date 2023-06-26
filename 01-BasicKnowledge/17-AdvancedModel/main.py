# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: main.py
@time: 2023/6/26 21:04
"""
import time
import hues
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss

# 数据路径
DATA_DIR = './data/'

# ToTensor
transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Train data and test data
train_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    transform=transformer,
    train=True,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    transform=transformer,
    train=False,
    download=True
)

# Data loader
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    drop_last=True
)

# VGG Model
vgg_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)


# Training function
def train(dataloader, model, loss_fn, optimizer, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)  # dataloader中的批量样本数
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                hues.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= num_batches
    correct /= size
    return train_loss, correct


# Training function
def test(dataloader, model, loss_fn, verbose=True, writer=None):
    size = len(dataloader.dataset)  # 样本总数
    num_batches = len(dataloader)  # dataloader中的批量样本数
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if verbose:
        hues.info(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct  # For reporting tuning results/ early stopping


if __name__ == '__main__':
    hues.info(f'The length of train data is {len(train_data)}, the length of test data is {len(test_data)}.')

    # Finding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Building model
    model = vgg_pretrained
    model.classifier.add_module('Linear', torch.nn.Linear(1000, 10))
    model.to(device)
    hues.info(model)

    # Loss function
    loss_fun = CrossEntropyLoss()
    loss_fun.to(device)

    # Optimizer
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(vgg_pretrained.parameters(), lr=learning_rate)

    # Using tensorboard to visualize network structure
    writer = SummaryWriter('./logs')

    # Running model
    epochs = 50
    for i in range(epochs):
        print(f"Epoch {i}\n-------------------------------")
        train_loss, train_cor = train(train_loader, model, loss_fun, optimizer)
        test_loss, test_cor = test(test_loader, model, loss_fun)
        writer.add_scalar('Loss/Train', train_loss, i)
        writer.add_scalar('Loss/Test', test_loss, i)
        writer.add_scalar('Accuracy/Train', train_cor, i)
        writer.add_scalar('Accuracy/Test', test_cor / len(test_data), i)

    hues.success("Done!")

    # Saving model
    timer = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    torch.save(model, f'./model/model_{timer}.pth')

    # Saving the structure of model
    X = None
    for X, y in test_loader:
        X = X.to(device)
        break
    writer.add_graph(vgg_pretrained, X)

    writer.close()  # Closing writer
    # tensorboard --logdir='logs'
