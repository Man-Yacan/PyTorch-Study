# -*- coding: utf-8 -*-

"""
@Author: Yacan Man
@Email: myxc@live.cn
@Website: https://www.manyacan.com
@software: PyCharm
@file: demo.py
@time: 2023/6/28 12:42
"""

if __name__ == '__main__':
    from rich.progress import track
    import time
    for i in track(range(10), description='Processing...'):
        time.sleep(1)
        print(i)

