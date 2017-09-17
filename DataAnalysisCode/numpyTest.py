#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: numpyTest.py
@time: 2017/9/17 
"""
import numpy as np


def func():
    pass


class Main():
    def __init__(self):
        #chapter1
        lst = [[1, 3, 5], [2, 4, 6]]
        print(type(lst))
        np_lst = np.array(lst, dtype=np.float)
        print(type(np_lst))
        # bool,int,int8,int16,int32,int64,int128,uint8,uint16,uint32,uint64,uint128,float,float16,float32,float64
        # complex64/128
        print(np_lst.shape)
        print(np_lst.ndim)
        print(np_lst.dtype)
        print(np_lst.itemsize)
        print(np_lst.size)

        #chapter2
        print(np.zeros([2, 4]))
        print(np.ones([3, 5]))
        print("Rand:")
        print(np.random.rand(2, 4))
        print(np.random.rand())
        print("Randint:")
        print(np.random.randint(1, 10, [3, 2]))
        print("Randn:")
        print(np.random.randn(2, 4))
        print("Choice:")
        print(np.random.choice([10, 20, 30]))
        print("Distribute:")
        print(np.random.beta(1, 10, 100))

        #chapter3 operation



if __name__ == '__main__':
    Main()
