# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: main.py
@time: 2023/6/24 10:16
"""
import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


demo = MyModule()
x = torch.tensor(1.)
print(demo(x))
