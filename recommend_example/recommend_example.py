#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: recommend_example.py
@time: 2017/9/19 
"""
import os
import math
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import pairwise
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

""" 基于http://python.jobbole.com/85516/ 整理"""


def predict(ratings, similarity, type="user"):
    """
    Description: 预测分数

    :param ratings: 评分矩阵
    :param similarity: 相似矩阵
    :param type:选择推荐类型
    :return:
    """
    if type == 'user':
        """
        mean
        经常操作的参数为axis，以m * n矩阵举例：
        axis 不设置值，对 m*n 个数求均值，返回一个实数
        axis = 0 ：压缩行，对各列求均值，返回 1* n 矩阵
        axis = 1 ：压缩列，对各行求均值，返回 m *1 矩阵
        """

        mean_user_rating = ratings.mean(axis=1)
        """
        a[:,np.newaxis]:能够将列表a每个元素变成一行
        """
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [(np.abs(similarity).sum(axis=1))]).T
    elif type == 'item':
        print ratings.shape
        print similarity.shape
        print np.array([np.abs(similarity).sum(axis=1)]).shape
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    """
    Description: 只用有评分的进行，误差分析
    :param prediction:
    :param ground_truth:
    :return:
    """

    # nonzero获取不为0的的坐标
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(prediction, ground_truth))


class Main():
    def __init__(self):
        # os.chdir("D:\\work\\liujm\\2017\\9\\20170919\\ml-20m\\ml-20m")
        os.chdir("D:\\work\\liujm\\2017\\9\\20170911\\ml-100k\\ml-100k")
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        # df = pd.read_csv(".\\ml-100k\u.data", sep="\t", names=header)
        df = pd.read_csv(".\\ml-100k\u.data", sep="\t", names=header)
        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]
        print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

        train_data, test_data = model_selection.train_test_split(df, test_size=0.2)

        # 生成评分矩阵
        train_data_matrix = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        test_data_matrix = np.zeros((n_users, n_items))
        for line in test_data.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        # 计算用户的余弦相似性(使用矩阵余弦距离)
        user_similarity = pairwise.cosine_similarity(train_data_matrix, dense_output=True)
        # 计算物品的余弦距离
        item_similarity = pairwise.cosine_similarity(train_data_matrix.T, dense_output=True)
        # 基于物品的推荐
        item_prediction = predict(train_data_matrix, item_similarity, type='item')
        # 基于用户的推荐
        user_prediction = predict(train_data_matrix, user_similarity, type='user')

        # 将nan值转化成0
        for i in range(len(item_prediction)):
            sample = item_prediction[i]
            for j in range(len(sample)):
                if np.isnan(sample[j]):
                    sample[j] = 0

        # 基于内存输出CF RMSE
        print 'User-based CF RMSE:' + str(rmse(user_prediction, test_data_matrix))
        print 'Item-based CF RMSE:' + str(rmse(item_prediction, test_data_matrix))

        # 计算Movielens数据集的稀疏度
        sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
        print 'The sparsity level of MovieLens100k is ' + str(sparsity * 100) + '%'

        # 将训练矩阵用SVD分解
        u, s, vt = svds(train_data_matrix, k=20)
        s_diag_matrix = np.diag(s)
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

        print 'User-based CF MSE:' + str(rmse(X_pred, test_data_matrix))


if __name__ == '__main__':
    Main()
