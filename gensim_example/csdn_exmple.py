#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: csdn_exmple.py
@time: 2017/9/18 
"""

from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm, metrics
import numpy as np
import os, re, time, logging, codecs
import jieba
import pickle as pkl

"""转自http://lib.csdn.net/article/machinelearning/43951"""

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )


class loadFolders(object):  # 迭代器
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):
                yield file_abspath


class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    content = readFile(file_path)
                    yield catg, content


"""
使用codecs 处理编码问题
"""


def saveFile(savepath, content):
    """
    :Descirption : 保存文件
    :param savepath: 保存路径
    :param content: 需要保存的内容
    :return:
    """
    fp = codecs.open(savepath, 'w+', 'gb18030')
    fp.write(content)
    fp.close()


def readFile(path):
    """
    Descriprtion: 读取文件
    :param path:
    :return:
    """
    fp = codecs.open(path, "r", 'gb18030', errors='ignore')
    content = fp.read()
    fp.close()
    return content


def contentSeg(content, cut_all=False):
    """
    Description: 内容分词
    :param content: 读取文件后的内容
    :param cut_all: 判断是否使用全分词
    :return:
    """
    content = content.replace("\r\n", "")  # 删除换行
    content = content.replace(" ", "")  # 删除空行、多余的空格
    content_seg = jieba.cut(content, cut_all=cut_all)  # 为文件内容分词
    return content_seg


def contentRm(content_seg):
    content_rm = []
    stop_words_list = getStopWords()
    for word in content_seg:
        if word not in stop_words_list:
            content_rm.append(word)
    return content_rm


def getDictionary(content_rm_list):
    """

    :param content_rm_list:
    :return:
    """
    return corpora.Dictionary(content_rm_list)


def getStopWords(path="D:/work/stopwords.txt"):
    """

    :param path:
    :return:
    """
    return set(readFile(path).split("\r\n"))


def svmClassify(train_set, train_tag, test_set, test_tag):
    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set, train_tag)
    train_pred = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)

    train_true_ratio = metrics.accuracy_score(train_tag, train_pred)
    test_true_ratio = metrics.accuracy_score(test_tag, test_pred)
    print('=== 分类训练完毕，分类结果如下 ===')
    print('训练集准确率: {e}'.format(e=train_true_ratio))
    print('检验集准确率: {e}'.format(e=test_true_ratio))

    return clf_res


def main():
    path_doc_root = "D:\\work\\Reduced"  # 根目录 即存放按类分类好的文本集
    path_tmp = "D:\\work\\tmp"  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsimodel = os.path.join(path_tmp, 'lsi_model.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')
    n = 1

    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    # # ===================================================================
    # # # # 第一阶段，  遍历文档，生成词典,并去掉频率较少的项
    #       如果指定的位置没有词典，则重新生成一个。如果有，则跳过该阶段

    content_rm_list = []
    if not os.path.exists(path_dictionary):
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        files = loadFiles(path_doc_root)
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                content = msg[1]
                content = contentSeg(content)
                content = contentRm(content)
                content_rm_list.append(content)
                if int(i / n) % 1000 == 0:
                    print('{t} *** {i} \t docs has been dealed'
                          .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

        dictionary = corpora.Dictionary(content_rm_list)
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)
        print('=== 词典已经生成 ===')
    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第二阶段，  开始将文档转化成tfidf
    if not os.path.exists(path_tmp_tfidf):
        print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        # 如果指定的位置没有tfidf文档，则生成一个。如果有，则跳过该阶段
        if not dictionary:  # 如果跳过了第一阶段,则从指定位置读取词典
            dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                content = msg[1]
                content = contentSeg(content)
                content = contentRm(content)
                content_bow = dictionary.doc2bow(content)
                content_tfidf = tfidf_model[content_bow]
                tmp = corpus_tfidf.get(catg, [])
                tmp.append(content_tfidf)
                if tmp.__len__() == 1:
                    corpus_tfidf[catg] = tmp

            if i % 10000 == 0:
                print('{i} files is dealed'.format(i=i))

        # 将tfidf中间结果存储起来
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg),
                                       corpus_tfidf.get(catg),
                                       id2word=dictionary
                                       )
            print('catg {c} has been transformed into tfidf vector'.format(c=catg))
        print('=== tfidf向量已经生成 ===')
    else:
        print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第三阶段，  开始将tfidf转化成lsi
    if not os.path.exists(path_tmp_lsi):
        print('=== 未检测到有lsi文件夹存在，开始生成lsi向量 ===')
        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf:
            print('--- 未检测到tfidf文档，开始从磁盘中读取 ---')
            # 从对应文件种读取所有类别
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if not t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg)
                corpus = corpora.MmCorpus(path)


if __name__ == '__main__':
    main()
