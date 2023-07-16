# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: demo.py
@time: 2023/7/12 16:05
"""


class D:
    # def _setattr__(self, name, value):
    #     self.__dict__[name] = value

    def _setattr__(self, name, value):
        self.name = value


if __name__ == '__main__':
    d = D()
    d.name = '小甲鱼'
    print(d.name)
