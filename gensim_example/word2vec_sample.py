#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: word2vec_sample.py
@time: 2017/9/14 
"""

from gensim.models.word2vec import Word2Vec
import gensim


def load_model():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        './GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
    return model


def test_model(model):
    pass


def func():
    pass


class Main():
    def __init__(self):
        model = load_model()
        print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


if __name__ == '__main__':
    Main()
