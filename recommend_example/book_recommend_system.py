#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: book_recommend_system.py
@time: 2017/9/20 
"""
import numpy as np
import pandas as pd
import collections
from scipy.stats import pearsonr
from sklearn.metrics import pairwise
from scipy.spatial.distance import cosine


def process_data():
    """
    Descrption:将数据转化成效用矩阵,去除掉不超过min_ratings个人评论的商品
    :return:
    """
    path = "D:\\work\\liujm\\2017\\9\\20170911\\ml-100k\\ml-100k\\u.data"
    save_path = "D:\\work\\recommend_system\\utilitymatrix.csv"
    df = pd.read_csv(path, sep='\t', header=None)
    df_info = pd.read_csv("D:\\work\\liujm\\2017\\9\\20170911\\ml-100k\\ml-100k\\u.item", sep="|", header=None)
    movie_list = [df_info[1].tolist()[idx] + ";" + str(idx + 1) for idx in xrange(len(df_info[1].tolist()))]

    min_ratings = 50
    n_users = len(df[0].drop_duplicates().tolist())
    n_movies = len(movie_list)
    movies_rated = list(df[1])
    counts = collections.Counter(movies_rated)
    dfout = pd.DataFrame(columns=['user'] + movie_list)

    tore_movie_list = []
    for i in xrange(1, n_users):
        tmp_movie_list = [0 for j in xrange(n_movies)]
        df_tmp = df[df[0] == i]
        for k in df_tmp.index:
            if counts[df_tmp.ix[k][1]] >= min_ratings:
                tmp_movie_list[df_tmp.ix[k][1] - 1] = df.ix[k][2]
            else:
                tore_movie_list.append(df.ix[k][1])

        dfout.loc[i] = [i] + tmp_movie_list

    tore_movie_list = list(set(tore_movie_list))
    dfout.drop(dfout.columns[tore_movie_list], axis=1, inplace=True)

    dfout.to_csv(save_path, index=None)


def read_data(path="D:\\work\\recommend_system\\utilitymatrix.csv"):
    """
    Description：读取数据为pandas类型,并设定默认读取路径
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    return df


def imputation(inp, Ri):
    """
    Description：未评分的使用替代评分策略，选项包括使用items的平均分和用户的平均分
    :param inp: 选择替代方法:useraverage/itemaverage
    :param Ri: 评分矩阵
    :return:
    """

    Ri = Ri.astype(float)

    def userav():
        """
        Description: 使用用户的已经评分的平均分数代替未评分的位置
        :return:
        """
        for i in xrange(len(Ri)):
            #
            Ri[i][Ri[i] == 0] = sum(Ri[i]) / float(len(Ri[i][Ri[i] > 0]))
        return Ri

    def itemav():
        """
        Description: 使用商品的评分分代替未评分的位置
        :return:
        """
        for i in xrange(len(Ri[0])):
            Ri[:, i][Ri[:, i] == 0] = sum(Ri[:, i]) / float(len(Ri[:, i][Ri[:, i] > 0]))
        return Ri

    switch = {'useraverage': userav(), 'itemaverage': itemav()}
    return switch[inp]


def matrix_sim(matrix, type="cosine"):
    """
    Descriptsion：计算矩阵相关性系数
    :param matrix: 输入的评分矩阵
    :param type: 相关性系数的选型
    :return:
    """
    if type == "cosine":  # 使用余弦距离计算矩阵
        return pairwise.cosine_similarity(matrix, dense_output=True)
    n_rows = len(matrix)
    cor_matrix = np.zeros((n_rows, n_rows))

    for u in xrange(n_rows):  # 使用皮尔逊相关系数计算相关性矩阵
        cor_matrix[u][u] = 1.0
        for v in xrange(u, n_rows):
            cor_matrix[u][v] = sim(matrix[u], matrix[v], metric="pearson")
            cor_matrix[v][u] = cor_matrix[u][v]
    return cor_matrix


def sim(x, y, metric='cosine'):
    """
    Description: 计算相似性
    :param x: 向量x
    :param y: 向量y
    :param metric: 判断使用哪种相似性
    :return:
    """
    if metric == 'cosine':
        return 1 - cosine(x, y)
    else:
        return pearsonr(x, y)[0]


class UserBasedCF(object):
    def __init__(self, data, K=10, type='cosine'):
        """
        Description: 进行用户协同过滤初始化
        n_items : 商品的数量
        :param data: 评分矩阵
        :param type: 相似性选型
        """

        self.data = data
        self.user_similarity = matrix_sim(self.data, type=type)
        self.n_items = len(data[0])
        self.K = K

    def get_k_neighs_per_user(self, u):
        """
        Description: 获取每个用户的最相近用户的K个用户列表
        作用: 1.为了压缩计算矩阵的维数，2.相似度不高的用户可能产生负作用
        :param u: 用户u
        :param K: 邻居数
        :return:neighs_ratings_matrix 邻居的评分矩阵;neighs_similarity:邻居相似性
        """
        neighs_ratings_matrix = np.zeros((self.K + 1, self.n_items))
        neighs_similarity = []
        key = [x for x in xrange(len(self.user_similarity[u]))]
        similarity = dict(zip(key, self.user_similarity[u].tolist()))
        # 第一行为用户本身的评分矩阵
        neighs_ratings_matrix[0] = self.data[u]
        # 第一个元素为1，表示用户自身相关性为1
        neighs_similarity.append(1)
        count = 1

        for k, v in sorted(similarity.items(), key=lambda item: -item[1])[:self.K]:  # 根据相似性排序前K个用户
            neighs_ratings_matrix[count] = self.data[k]  # 将用户k的评分填充到邻居评分矩阵
            neighs_similarity.append(self.user_similarity[u].tolist()[k])  # 将相似性填充到邻居相似性列表
            count += 1
        return neighs_ratings_matrix, neighs_similarity

    def user_based_recommend(self, u, top=10):
        """
        Descriptsion:获取某个用户的前top个商品推荐列表
        :param u:用户u
        :param K: 使用K个邻居计算商品分数
        :param top: 商品推荐数量
        :return:recommend_dict : 返回推荐商品字典
        """

        # 获取邻居矩阵，获取邻居相似性列表
        neighs_ratings_matrix, neighs_similarity = self.get_k_neighs_per_user(u)
        recommend_dict = dict()
        # axis=1 按照行(用户)来计算均值
        mean_user_rating = neighs_ratings_matrix.mean(axis=1)
        # 评分矩阵减去均值
        ratings_diff = (neighs_ratings_matrix - mean_user_rating[:, np.newaxis])
        # 矩阵乘法，计算用户u对所用items的评分
        ratings = sum(self.data[u]) / len(self.data[u] > 0) + np.array(neighs_similarity).dot(
            ratings_diff) / sum(neighs_similarity)
        keys = [x for x in xrange(len(ratings))]
        key_rating = dict(zip(keys, ratings))
        cnt = 0
        # 遍历top推荐列表
        for k, rating in sorted(key_rating.items(), key=lambda item: -item[1]):
            if cnt < top and self.data[u][k] == 0:  # 需要评分等于0,即没有评分的商品
                recommend_dict.get(k + 1, None)
                recommend_dict[k + 1] = rating
                cnt += 1
            elif cnt == top:
                break
        return recommend_dict

    def users_based_recommend(self):
        """
        Description: 完成整体推荐字典
        :return:
        """
        recommend_dict_all = dict()
        for u in xrange(len(self.data)):
            recommend_dict_all[u] = self.user_based_recommend(u)
        return recommend_dict_all


class ItemBasedCF(object):
    def __init__(self, data, K=10, type='cosine'):
        """
        Description: 进行商品协同过滤初始化
        n_users : 商品的数量
        :param data: 评分矩阵
        :param type: 相似性选型
        """

        self.data = data
        self.item_similarity = matrix_sim(data.T, type=type)
        self.n_users = len(self.data)
        self.K = K

    def item_neighs_modify_item_similarity(self):
        for i in xrange(len(self.item_similarity)):
            items = np.argsort(self.item_similarity[i])[::-1]  # 根据相似度逆序返回商品id列表
            items = items[items != i]  # 去掉本身
            self.item_similarity[i, i] = 0.0
            for j in xrange(i, len(self.item_similarity)):
                if j not in items:  # triangular matrix
                    self.item_similarity[i, j] = 0.0
                    self.item_similarity[j, i] = 0.0

    def items_based_recommend(self, top=10):
        pred = self.data.mean(axis=0)[:, np.newaxis].T + self.data.dot(
            self.item_similarity) / np.array([np.abs(self.item_similarity).sum(axis=1)])
        recommend_dict_all = dict()
        for u in xrange(len(self.data)):
            items_idx = np.argsort(pred[u])[::-1]
            recommend_dict = dict()
            cnt = 0
            for i in items_idx:
                if self.data[u, i] == 0 and cnt < top:
                    recommend_dict[i+1] = pred[u, i]
                    cnt += 1
                elif cnt == top:
                    break
            recommend_dict_all[u] = recommend_dict

        return recommend_dict_all


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    import os

    os.chdir("D:\\work\\liujm\\2017\\9\\20170911\\ml-100k\\ml-100k")
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    # df = pd.read_csv(".\\ml-100k\u.data", sep="\t", names=header)
    df = pd.read_csv(".\\u.data", sep="\t", names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

    # 生成评分矩阵
    data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = line[3]
    import time

    # print "time.time(): %f " % time.time()
    # dm = UserBasedCF(data_matrix, K=50)
    # recommend_dict_all = dm.users_based_recommend()
    # print "time.time(): %f " % time.time()
    # for k, v in recommend_dict_all.iteritems():
    #     print k, "=>", v
    print "time.time(): %f " % time.time()
    dm = ItemBasedCF(data_matrix, K=50)
    recommend_dict_all = dm.items_based_recommend()
    print "time.time(): %f " % time.time()
    for k, v in recommend_dict_all.iteritems():
        print k, "=>", v

