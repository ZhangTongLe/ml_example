#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: tf_idf_example.py
@time: 2017/9/14 
"""

# 加载库
import os
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def func():
    pass


class Main():
    def __init__(self):
        # 设置当前工作路径
        os.chdir("D:/work/liujm/2017/9/20170901/sklearn_exercise")
        # 加载数据

        twenty_train = datasets.load_files("data/20news-bydate/20news-bydate-train")
        twenty_test = datasets.load_files("data/20news-bydate/20news-bydate-test")

        # 计算词频
        count_vect = CountVectorizer(stop_words="english", decode_error="ignore")
        x_train_counts = count_vect.fit_transform(twenty_train.data)

        # 使用TF-IDF进行特征提取
        tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
        x_train_tf = tf_transformer.transform(x_train_counts)

        # 词频表达乘TF-IDF
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)


if __name__ == '__main__':
    Main()
