#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: kerasTest.py
@time: 2017/9/17 
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import theano


def func():
    pass


class Main():
    def __init__(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        print(iris["target"])
        from sklearn.preprocessing import LabelBinarizer
        print(LabelBinarizer().fit_transform(iris["target"]))
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.25)

        labels_train = LabelBinarizer().fit_transform(train_target)
        labels_test = LabelBinarizer().fit_transform(test_target)

        model = Sequential(
            [
                Dense(5, input_dim=4),
                Activation("relu"),
                Dense(3),
                Activation("sigmoid"),

            ]
        )

        # model = Sequential()
        # model.add(Dense(5,input=4))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
        model.compile(optimizer=sgd, loss="categorical_crossentropy")
        model.fit(train_data, labels_train, epochs=200, batch_size=40)

        print(model.predict_classes(test_data))


if __name__ == '__main__':
    Main()
