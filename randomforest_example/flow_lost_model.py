#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: flow_lost_model.py
@time: 2017/9/14 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics




def data_loading():
    headers = ['phoneno', 'month', 'operators', 'ip', 'prov_city', 'publicchannel', 'port', 'version', 'ishigh_count',
               'islow_count', 'ishigh', 'islow', 'province_id', 'city_id', 'lottery', 'auction_ticket', 'public_num',
               'purchase_ticket', 'feedback', 'flow_activity', 'sign', 'sale_ticket', 'query_flow', 'check_bill',
               'modify_alert', 'guess', 'purchase_coin', 'crazy_puzzle', 'flow_junction', 'fee_exchange',
               'application_act', 'flow_exchange', 'update_check', 'give_flow', 'logout', 'wifi_exchange',
               'open_client', 'service', 'login', 'message', 'cattle_coin', 'coinchange', 'ticketchange',
               'experiencechange', 'first_day', 'last_day', 'last_day_to_end', 'during_days', 'appear_days',
               'lost_label'
               ]

    na_str_list = ["-", "None", "null", "NULL"]
    data = pd.read_table('data/sample.txt',
                         dtype={'publicchannel': str, 'phoneno': str, 'ip': str, 'version': str,
                                'prov_city': str, 'finace_level': str, 'city_id': str},
                         names=headers,
                         na_values=na_str_list)

    return data


def data_processing(data):
    data = data.fillna('-10')
    # 构建训练集和测试集
    x_y_train = data[data.month < 201603]
    x_y_test = data[data.month == 201603]

    # 扔掉的无用变量（增加时间信息）
    drop_list_all = ["phoneno", "month", "ip", "publicchannel", "version", "lost_label"]
    # 扔掉的无用变量（无时间信息）
    drop_list_no_time = ["phoneno", "month", "ip", "publicchannel", "version", "lost_label", 'first_day',
                         'last_day', 'last_day_to_end', 'during_days', 'appear_days']

    # 训练集
    x_train_all = x_y_train.drop(drop_list_all, axis="columns")
    x_train_no_time = x_y_train.drop(drop_list_no_time, axis="columns")

    # 测试集
    x_test_all = x_y_test.drop(drop_list_all, axis="columns")
    x_test_no_time = x_y_test.drop(drop_list_no_time, axis="columns")

    y_train = x_y_train.lost_label.values
    y_test = x_y_test.lost_label.values

    return x_train_all, x_test_all, x_train_no_time, x_test_no_time, y_train, y_test


def random_train(x_train_all, x_train_no_time, y_train):
    # 训练模型

    clf_all = RandomForestClassifier(n_jobs=2, max_features="auto", n_estimators=60, max_depth=None
                                     , oob_score=True, min_samples_leaf=1000, verbose=True)

    clf_no_time = RandomForestClassifier(n_jobs=2, max_features="auto", n_estimators=60, max_depth=None
                                         , oob_score=True, min_samples_leaf=1000, verbose=True)
    print clf_all
    print clf_no_time

    model_all = clf_all.fit(x_train_all, y_train.T)
    model_no_time = clf_no_time.fit(x_train_no_time, y_train.T)

    return model_all, model_no_time


def random_predict_and_evaluate(model_all, model_no_time, x_test_all, x_test_no_time):
    predicted_all = model_all.predict(x_test_all)
    predicted_no_time = model_no_time.predict(x_test_no_time)

    prob_all = model_all.predict_proba(x_test_all)[:, 1]
    prob_no_time = model_no_time.predict_proba(x_test_no_time)[:, 1]



def set_ch():
    """
    Description: 设置输出图片的字体、 解决保存图像是负号'-'显示为方块的问题
    :return: 
    """
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
