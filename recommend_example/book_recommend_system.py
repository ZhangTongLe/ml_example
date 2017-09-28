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
import copy
import time
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


def ratings_matrix(dataframe):
    """
    Description:评分记录转化成评分矩阵
    :param dataframe:
    :return:
    """
    n_users = dataframe.user_id.unique().shape[0]
    n_items = dataframe.item_id.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

    # 生成评分矩阵
    data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = line[3]

    return data_matrix


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
    def __init__(self, data, K=-1, type='cosine'):
        """
        Description: 进行用户协同过滤初始化
        n_items : 商品的数量
        :param data: 评分矩阵
        :param type: 相似性选型
        """

        self.data = data
        self.user_similarity = matrix_sim(data, type=type)
        self.n_items = len(data[0])
        self.n_uers = len(data)
        self.K = K

    def user_neighs_modify_item_similarity(self):
        """
        Descrption: 根据参数K，选择相似度最高的K个邻居,生成相似度矩阵,邻居的相似度为0.0
        :return:
        """
        user_neighs_similarity = np.zeros((self.n_uers, self.n_uers))
        for i in xrange(self.n_uers):
            items = np.argsort(self.user_similarity[i])[::-1][:self.K]  # 根据相似度逆序返回商品id列表
            items = items[items != i]  # 去掉本身
            for j in xrange(self.n_uers):
                if j in items:  # triangular matrix
                    user_neighs_similarity[i, j] = self.user_similarity[i, j]
        self.user_similarity = user_neighs_similarity

    def users_based_recommend(self, top=10):
        """

        :param top:
        :return:
        """
        if self.K != -1:
            self.user_neighs_modify_item_similarity()
        mean_user_rating = self.data.sum(axis=1, dtype=float) / np.count_nonzero(self.data, axis=1)
        rating_diff = self.data - mean_user_rating[:, np.newaxis]
        pred = mean_user_rating[:, np.newaxis] + self.user_similarity.dot(
            rating_diff) / np.array([np.abs(self.user_similarity).sum(axis=1)]).T
        recommend_dict_all = dict()
        for u in xrange(len(self.data)):
            items_idx = np.argsort(pred[u])[::-1]
            recommend_dict = dict()
            cnt = 0
            for i in items_idx:
                if self.data[u, i] == 0 and cnt < top:
                    recommend_dict[i + 1] = pred[u, i]
                    cnt += 1
                elif cnt == top:
                    break
            recommend_dict_all[u] = recommend_dict

        print "max:", pred.max()
        return recommend_dict_all


class ItemBasedCF(object):
    def __init__(self, data, K=-1, type='cosine'):
        """
        Description: 进行商品协同过滤初始化
        n_users : 商品的数量
        :param data: 评分矩阵
        :param type: 相似性选型
        """

        self.data = data
        self.item_similarity = matrix_sim(data.T, type=type)
        self.n_users = len(self.data)
        self.n_items = len(self.data[0])
        self.K = K

    def item_neighs_modify_item_similarity(self):
        """
        Descrption:
        :return:
        """
        item_neighs_similarity = np.zeros((self.n_items, self.n_items))
        for i in xrange(self.n_items):
            items = np.argsort(self.item_similarity[i])[::-1][:self.K]  # 根据相似度逆序返回商品id列表
            items = items[items != i]  # 去掉本身
            for j in xrange(self.n_items):
                if j in items:  # triangular matrix
                    item_neighs_similarity[i, j] = self.item_similarity[i, j]
        self.item_similarity = item_neighs_similarity

    def items_based_recommend(self, top=10):
        if self.K != -1:
            self.item_neighs_modify_item_similarity()
        pred = self.data.dot(self.item_similarity) / np.array([np.abs(self.item_similarity).sum(axis=1)])
        recommend_dict_all = dict()
        for u in xrange(len(self.data)):
            items_idx = np.argsort(pred[u])[::-1]
            recommend_dict = dict()
            cnt = 0
            for i in items_idx:
                if self.data[u, i] == 0 and cnt < top:
                    recommend_dict[i + 1] = pred[u, i]
                    cnt += 1
                elif cnt == top:
                    break
            recommend_dict_all[u] = recommend_dict

        print "max:", pred.max()
        return recommend_dict_all


class SlopeOne(object):
    """
    Description : 最简单的基于商品的协同过滤
    """

    def __init__(self, data):
        self.n_users = len(data)
        self.n_items = len(data[0])
        self.difmatrix = np.zeros((self.n_items, self.n_items))
        self.nratings = np.zeros((self.n_items, self.n_items))
        self.data = data

    def build_matrix(self):
        """
        Description: 建立相似性矩阵
        :return:
        """

        n_items = self.n_items
        n_users = self.n_users
        for i in xrange(n_items):
            for j in xrange(i + 1, n_items):
                n_counts = 0
                diff = 0
                for k in xrange(n_users):
                    if self.data[k, i] > 0 and self.data[k, j]:
                        n_counts += 1
                        diff += (self.data[k, i] - self.data[k, j])
                self.difmatrix[i, j] = (diff + 1) / (n_counts + 1)
                self.difmatrix[j, i] = self.difmatrix[i, j]
                self.nratings[i, j] = n_counts
                self.nratings[j, i] = self.nratings[i, j]

    def slop_one_recommend(self, K=20):
        """
        Description: 商品推荐
        :param K:
        :return:
        """
        self.build_matrix()
        pred = np.zeros((self.n_users, self.n_items))
        for u in xrange(self.n_users):
            for m in xrange(self.n_items):
                if self.data[u, m] == 0:
                    pred[u, m] = np.dot(self.data[u] + self.difmatrix[m], self.nratings[m]) / self.nratings[m].sum()
        return pred


class ModelCF(object):
    """
    Description: 基于模型的协同过滤
    """

    def __init__(self, Umatrix):
        self.Umatirx = Umatrix

    def SGD(self, K, iterations=3, alpha=1, l=0.1, tol=0.001):
        """
        Descrption: Stochastic Gradient Descent 随机梯度下降
        :param K: 特征的数量
        :param iterations: 迭代次数
        :param alpha: 学习速率
        :param l: 正则化系数,防止过拟合
        :param tol: 收敛判据 convergence tolerance
        :return:
        """
        matrix = self.Umatirx
        # 获取矩阵的行数
        n_rows = len(matrix)
        # 获取矩阵的列数
        n_cols = len(matrix)
        # 生成两个随机矩阵
        P = np.random.rand(n_rows, K)
        Q = np.random.rand(n_cols, K)
        Qt = Q.T
        cost = -1
        for it in xrange(iterations):
            for i in xrange(n_rows):
                for j in xrange(n_cols):
                    if matrix[i][j] > 0:
                        # 误差
                        eij = matrix[i][j] - np.dot(P[i, :], Qt[:, j])
                        for k in xrange(K):
                            P[i][k] += np.round(alpha * (2 * eij * Qt[k][j] - l * P[i][k]), 0)
                            Qt[k][j] += np.round(alpha * (2 * eij * P[i][k] - l * Qt[k][j]), 0)
            cost = 0
            for i in xrange(n_rows):
                for j in xrange(n_cols):
                    if matrix[i][j] > 0:
                        cost += pow(matrix[i][j] - np.dot(P[i, :], Qt[:, j]), 2)
                        for k in xrange(K):
                            cost += l * (pow(P[i, k], 2) + pow(Qt[k, j], 2))

            print "第" + str(it) + "迭代,cost:" + str(round(cost, 0))
            # alpha = alpha * 0.9
            if cost < tol:
                break

        return np.round(np.dot(P, Qt), 0)

    def ALS(self, K, iterations=3, l=0.001, tol=0.001):
        """
        Description: Alternating Least Square ALS 交替最小二乘法
        通常没有SGD/SVD精确,但是速度较快，易用于并行计算
        :param K: 特征维数
        :param iterations: 迭代次数
        :param l: 正则化系数
        :param tol: 收敛判据
        :return:
        """
        matrix = self.Umatirx

        n_rows = len(matrix)
        n_cols = len(matrix[0])
        P = np.random.rand(n_rows, K)
        Q = np.random.rand(n_cols, K)
        Qt = Q.T
        err = 0.
        matrix = matrix.astype('float')
        mask = matrix > 0
        mask[mask == True] = 1
        mask[mask == False] = 0
        mask = mask.astype(np.float64, copy=False)
        for it in xrange(iterations):
            for u, mask_u in enumerate(mask):
                P[u] = np.linalg.solve(np.dot(Qt, np.dot(np.diag(mask_u), Q)) + l * np.eye(K),
                                       np.dot(Qt, np.dot(np.diag(mask_u), matrix[u].T))).T

            for i, mask_i in enumerate(mask.T):
                Qt[:, i] = np.linalg.solve(np.dot(P.T, np.dot(np.diag(mask_i), P)) + l * np.eye(K),
                                           np.dot(P.T, np.dot(np.diag(mask_i), matrix[:, i])))

            err = np.sum((mask * (matrix - np.dot(P, Qt))) ** 2)
            if err < tol:
                break
            print "第" + str(it + 1) + "迭代,cost:" + str(round(err, 0))

        return np.round(np.dot(P, Qt), 0)

    def NMF_alg(self, K, inp='none', l=0.001):
        """
        Description: Non-negative Matrix Factorization 非负矩阵分解
        :param K: 特征维度
        :param inp: 缺失值替换方法
        :param l: 正则化系数
        :return:
        """
        from sklearn.decomposition import NMF
        matrix = self.Umatirx
        R_tmp = copy.copy(matrix)
        R_tmp = R_tmp.astype(float)
        # inputation
        if inp != 'none':
            R_tmp = imputation(inp, matrix)
        nmf = NMF(n_components=K, alpha=l)
        P = nmf.fit_transform(R_tmp)
        R_tmp = np.dot(P, nmf.components_)
        return R_tmp

    def SVD(self, K, inp='none'):
        """
        Description: Singular Value Decomposition 奇异值分解
        :param K: 特征
        :param inp: 缺失值替代方法
        :return:
        """

        from sklearn.decomposition import TruncatedSVD
        matrix = self.Umatirx
        R_tmp = copy.copy(matrix)
        R_tmp = R_tmp.astype(float)
        # inputation
        if inp != 'none':
            R_tmp = imputation(inp, matrix)

        mean_user_rating = R_tmp.sum(axis=1, dtype=float) / np.count_nonzero(R_tmp, axis=1)
        rating_diff = R_tmp - mean_user_rating[:, np.newaxis]
        svd = TruncatedSVD(n_components=K, random_state=4)
        R_k = svd.fit_transform(rating_diff)
        R_tmp = svd.inverse_transform(R_k)
        R_tmp = mean_user_rating[:, np.newaxis] + R_tmp

        return np.round(R_tmp, 0)

    def SVD_EM(self, K, inp='none', iterations=1000, tol=0.001):
        """
        Description : SVD+最大期望算法
        :param K: 特征维数
        :param inp: 缺失值替代方法
        :param iterations: 迭代次数
        :param tol: 收敛判据
        :return:
        """

        from sklearn.decomposition import TruncatedSVD
        matrix = self.Umatirx
        R_tmp = copy.copy(matrix)
        n_rows = len(matrix)
        n_cols = len(matrix[0])
        # inputation
        if inp != 'none':
            R_tmp = imputation(inp, matrix)

        # define svd
        svd = TruncatedSVD(n_components=K, random_state=4)
        err = -1
        for it in xrange(iterations):
            # m-step
            R_k = svd.fit_transform(R_tmp)
            R_tmp = svd.inverse_transform(R_k)
            # e-step and error evaluation
            err = 0
            for i in xrange(n_rows):
                for j in xrange(n_cols):
                    if matrix[i][j] > 0:
                        err += pow(matrix[i][j] - R_tmp[i][j], 2)
                        R_tmp[i][j] = matrix[i][j]

            print "第" + str(it + 1) + "迭代,cost:" + str(round(err, 0))
            if err < tol:
                print it, 'tol reached'
                break
        return np.round(R_tmp, 0)


class CBF(object):
    """
    Descrption:从描述商品的数中抽取用户特征
    Content-based Filtering 基于内容的过滤
    """

    def __init__(self, data, movies):
        self.ratings_matrix = data.astype('float')
        self.n_features = len(movies[0])
        self.movies = movies

    def CBF_Average(self):
        mean_user_rating = self.ratings_matrix.mean(axis=1, dtype=float)
        ratings_diff = self.ratings_matrix - mean_user_rating[:, np.newaxis]
        V = np.dot(ratings_diff, self.movies) / self.movies.sum(axis=0, dtype=float)[np.newaxis]
        pred = np.dot(V, self.movies.T)
        return pred

    def CBF_regression(self, alpha=0.01, l=0.0001, its=50, tol=0.001):
        n_features = self.n_features + 1
        n_users = len(self.ratings_matrix)
        n_items = len(self.ratings_matrix[0])
        movies_feats = np.ones((n_items, n_features))
        movies_feats[:, 1:] = self.movies
        movies_feats = movies_feats.astype('float')

        p_matrix = np.random.rand(n_users, n_features)
        p_matrix[:, 0] = 1.
        err = 0
        cost = -1
        for it in xrange(its):
            print 'it:', it, ' -- ', cost
            for u in xrange(n_users):
                for f in xrange(n_features):
                    if f == 0:
                        for m in xrange(n_items):
                            if self.ratings_matrix[u, m] > 0:
                                diff = np.dot(p_matrix[u], movies_feats[m]) - self.ratings_matrix[u, m]
                                p_matrix[u, f] += - alpha * (diff * movies_feats[m][f])
                    else:
                        for m in xrange(n_items):
                            if self.ratings_matrix[u, m] > 0:
                                diff = np.dot(p_matrix[u], movies_feats[m]) - self.ratings_matrix[u, m]
                                p_matrix[u, f] += -alpha * (diff * movies_feats[m][f]) + l * p_matrix[u, f]
            cost = 0
            for u in xrange(n_users):
                for m in xrange(n_items):
                    if self.ratings_matrix[u][m] > 0:
                        cost += 0.5 * pow(self.ratings_matrix[u, m] - np.dot(p_matrix[u], movies_feats[m]), 2)
                for f in xrange(1, n_features):
                    cost += float(1 / 2.0) * pow(p_matrix[u][f], 2)
            print 'err', cost
            if cost < tol:
                print 'err', cost
                break
        return np.dot(p_matrix, movies_feats)


class AssociationRules(object):
    """
    Description: 关联规则
    """

    def __init__(self, Umatrix, Movieslist, min_support=0.1, min_confidence=0.1, likethreshold=3):
        """
        Description: 关联规则初始化
        :param Umatrix: 评分矩阵
        :param Movieslist: 电影列表
        :param min_support: 支持度
        :param min_confidence: 置信度
        :param likethreshold: 下限的过滤分数
        """
        self.min_support = min_support  # 支持度
        self.min_confidence = min_confidence  # 置信度
        self.Movieslist = Movieslist  # 电影清单
        n_items = len(Umatrix[0])
        transactions = []  # 项集
        for u in Umatrix:
            # 评分>likethreshold  才能构成项集
            s = [i for i in xrange(len(u)) if u[i] > likethreshold]
            if len(s) > 0:
                transactions.append(s)
        # 将所有的item展开成一行
        flat = [item for sublist in transactions for item in sublist]
        # 初始化的items
        inititems = map(frozenset, [[item] for item in frozenset(flat)])
        # 将项集转化成无需集合
        set_trans = map(set, transactions)
        # 过滤出在关联规则组合中出现的元素
        sets_init, self.dict_sets_support = self.filterSet(set_trans, inititems)
        # 推荐系统只需要两项关联规则
        setlen = 2
        # 构建所有可能出现的组合
        items_temp = self.combine_lists(sets_init, setlen)
        # 过滤出所有的频繁集和支持度
        self.freq_sets, sup_tmp = self.filterSet(set_trans, items_temp)
        # 更新支持度集合
        self.dict_sets_support.update(sup_tmp)
        # 关联规则置信度矩阵初始化
        self.ass_matrix = np.zeros((n_items, n_items))
        # 构建关联规则置信度矩阵，遍历频繁集
        for freqset in self.freq_sets:
            list_setitems = [frozenset([item]) for item in freqset]
            self.calc_confidence_matrix(freqset, list_setitems)

    def filterSet(self, set_trans, likeditems):
        """
        Description: 过滤出组合中出现的元素
        :param set_trans: 给定的项集
        :param likeditems: 所有的元素
        :return:
        """
        itemscnt = {}
        # 遍历给定同时出现的项集
        for id in set_trans:
            # 遍历所有可能的元素
            for item in likeditems:
                # 如果某一个元素出现在某一个项里面
                if item.issubset(id):
                    # 统计元素出现的次数
                    itemscnt.setdefault(item, 0)
                    itemscnt[item] += 1
        # 计算多少个项集合
        num_items = float(len(set_trans))
        # 频繁集
        freq_sets = []
        # 支持度集
        dict_sets = {}
        # 遍历每个元素出现的次数
        for key in itemscnt:
            # 计算key 对应的支持度
            support = itemscnt[key] / num_items
            if support >= self.min_support:  # 如果支持度大于设定的支持度
                freq_sets.insert(0, key)
            # 插入支持度
            dict_sets[key] = support
        return freq_sets, dict_sets

    def combine_lists(self, freq_sets, setlen):
        """
        Description: 寻找所有的可能, 当setlen=2，寻找可能同时出现的两个商品组合
        :param freq_sets: 遍历的商品组合，setlen=2 的时候，就是单个元素集合
        :param setlen: 可能的组合长度
        :return:
        """
        set_items_list = []
        n_sets = len(freq_sets)
        for i in xrange(n_sets):
            for j in xrange(i + 1, n_sets):
                set_list1 = list(freq_sets[i])[:setlen - 2]
                set_list2 = list(freq_sets[j])[:setlen - 2]
                if set(set_list1) == set(set_list2):
                    # 计算并集 union
                    set_items_list.append(freq_sets[i].union(freq_sets[j]))
        return set_items_list

    def calc_confidence_matrix(self, freqset, list_setitems):
        """
        Description: 计算相似性矩阵
        :param freqset: 某一项频繁集
        :param list_setitems: 频繁集内部的单个元素项的列表
        :return:
        """
        # 遍历推荐的商品(traget:目标)
        for target in list_setitems:
            # self.dict_sets_support[freqset]同时出现的项集的支持度
            # self.dict_sets_support[freqset - target] 已经打分的项集的支持度
            # 计算基于已经打分的商品的支持度下，推荐商品的置信度
            confidence = self.dict_sets_support[freqset] / self.dict_sets_support[freqset - target]
            # 大于最低的置信度
            if confidence > self.min_confidence:
                self.ass_matrix[list(freqset)[0]][list(target)[0]] = confidence

    def GetRecItems(self, u_vec, indxs=False):
        """
        Description: 计算某个向量的推荐的商品列表
        :param u_vec: 给定的用户
        :param indxs: false过滤掉已经看过的商品
        :return:
        """
        vec_recs = np.dot(u_vec, self.ass_matrix)
        sortedweight = np.argsort(vec_recs)
        seenindxs = [indx for indx in xrange(len(u_vec)) if u_vec[indx] > 0]
        seenmovies = np.array(self.Movieslist)

        recitems = np.array(self.Movieslist)[sortedweight]
        recitems = [m for m in recitems if m in seenmovies]
        if indxs:
            vec_recs[seenindxs] = -1
            recsvec = np.argsort(vec_recs)[::-1][np.argsort(vec_recs) > 0]
            return recsvec
        return recitems[::-1]


class Main:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def test_model(self):
        dm = ModelCF(self.data_matrix)
        pred = dm.SVD_EM(K=10)
        mask = data_matrix > 0
        mask[mask == True] = 1
        mask[mask == False] = 0
        err = np.sum((mask * (self.data_matrix - pred) ** 2))
        print err

    def test_cf(self):
        print "time.time(): %f " % time.time()
        dm = UserBasedCF(self.data_matrix)
        recommend_dict_all = dm.users_based_recommend(top=20)
        print recommend_dict.get(0, [])
        print "time.time(): %f " % time.time()


if __name__ == '__main__':
    import os

    os.chdir("D:\\work\\liujm\\2017\\9\\20170911\\ml-100k\\ml-100k")
    # os.chdir("D:\\work\\liujm\\2017\\9\\20170919\\ml-20m\\ml-20m")
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    # df = pd.read_csv(".\\ml-100k\u.data", sep="\t", names=header)
    # df = pd.read_csv(".\\ratings.csv", sep=',', names=header)
    df = pd.read_csv(".\\u.data", sep="\t", names=header)
    data_matrix = ratings_matrix(df)

    # df_info_header = ['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
    #                   'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    #                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    #
    # df_info = pd.read_csv('.\\u.item', sep='|', names=df_info_header)
    #
    # moviescats = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
    #               'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    #               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    # dfout_movies = pd.DataFrame(columns=['movie_id'] + moviescats)
    # startcatsindx = 5
    #
    # # matrix movies's content
    # cnt = 0
    # movies_list = df_info.movie_id.unique()
    # n_movies = len(movies_list)
    # n_features = len(moviescats)
    # content_matrix = np.zeros((n_movies, n_features))
    # for x in xrange(n_movies):
    #     content_matrix[x] = df_info.iloc[x][startcatsindx:].tolist()
    #
    # print content_matrix.shape
    #
    # cbf = CBF(data_matrix, content_matrix)
    # pred = cbf.CBF_regression()
    # # keys = [x for x in xrange(n_movies)]
    #
    # items_idx = np.argsort(pred[1])[::-1]
    # recommend_dict = dict()
    # cnt = 0
    # top = 10
    # for i in items_idx:
    #     if data_matrix[1, i] == 0 and cnt < top:
    #         recommend_dict[i + 1] = pred[1, i]
    #         cnt += 1
    #     elif cnt == top:
    #         break
    # # recommend_dict_all[u] = recommend_dict
    # print recommend_dict

    so = SlopeOne(data_matrix)
    pred = so.slop_one_recommend()
    # keys = [x for x in xrange(n_movies)]

    items_idx = np.argsort(pred[1])[::-1]
    recommend_dict = dict()
    cnt = 0
    top = 10
    for i in items_idx:
        if data_matrix[1, i] == 0 and cnt < top:
            recommend_dict[i + 1] = pred[1, i]
            cnt += 1
        elif cnt == top:
            break
    # recommend_dict_all[u] = recommend_dict
    print recommend_dict


    # mask = data_matrix > 0
    # mask[mask == True] = 1
    # mask[mask == False] = 0
    # err = np.sum((mask * (data_matrix - pred) ** 2))
    # print err
