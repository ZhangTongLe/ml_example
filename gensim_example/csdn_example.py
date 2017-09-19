#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: csdn_example.py
@time: 2017/9/18 
"""

from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn import svm, metrics
import numpy as np
import os, re, time, logging, codecs
import jieba
import pickle as pkl

"""
代码转自http://lib.csdn.net/article/machinelearning/43951,根据个人喜好进行的细微修改

"""

# 日志输出格式
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )


class loadFolders(object):  # 迭代器
    """
    Description:使用迭代器来载入文件夹,能够减少内存消耗
    """

    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):
                yield file_abspath


class loadFiles(object):
    """
    Description:使用迭代器来载入文件, 返回类别和文件内容
    备注: 类别是每个文件夹名字
    """

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
使用codecs 处理编码问题，比较安全的使用中文编码
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
    """
    Description: 删除常用停用词
    :param content_seg:分词后的列表
    :return:删除停用词后的列表
    """
    content_rm = []
    stop_words_list = getStopWords()
    for word in content_seg:
        if word not in stop_words_list:
            content_rm.append(word)
    return content_rm


def getDictionary(content_rm_list):
    """
    Description: 获取字典
    :param content_rm_list:
    :return:
    """
    return corpora.Dictionary(content_rm_list)


def getStopWords(path="D:/work/stopwords.txt"):
    """
    Description: 获取停用词列表
    :param path:
    :return:
    """
    return set(readFile(path).split("\r\n"))


def svmClassify(train_set, train_tag, test_set, test_tag):
    """
    Description: 建立分类器
    :param train_set: 训练集特征集
    :param train_tag: 训练集标签集
    :param test_set: 测试集特征集
    :param test_tag: 测试集标签集合
    :return: 返回一个分类器
    """
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
    """
    Descption: 全流程测试
    1.生成词典：
    中文分词(jieba) =>
    去除停用词 =>
    将停用词列表输入到dictionary中,生成词典 =>
    保存词典路径
    2.根据词典,将文档转换成tfidf:
    中文分词(jieba) =>
    去除停用词=>
    使用字典(doc2vec)对每一篇文章生成词袋 =>
    使用TfidfModel将每个词袋转换成tfidf向量 =>
    分类别保存每一个tfidf矩阵
    3.根据词典，将tfidf转换成lsi:
    依据词典和tfidf矩阵，使用LsiModel生成lsi向量
    4.将lsi处理后，并作为特征输入到svm，训练分类器:
    读取lsi文档 =>
    生成lsi矩阵 (csr_matrix) =>
    生成测试集和训练集 =>
    进行分类
    5.测试，使用新的文本进行分类:
    分词 => 去除停用词 => 词袋 => tfidf向量 => lsi向量 => lsi矩阵 =>分类器 => 输出结果
    :return:
    """

    # 定义中间会使用的路径
    path_doc_root = "D:\\work\\Reduced"  # 根目录 即存放按类分类好的文本集
    path_tmp = "D:\\work\\tmp"  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsimodel = os.path.join(path_tmp, 'lsi_model.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')
    n = 1  # 文档采样频率，n代表每n个采样1个

    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    """
     ===================================================================
            第一阶段，  遍历文档，生成词典,并去掉频率较少的项
            如果指定的位置没有词典，则重新生成一个。如果有，则跳过该阶段
    """

    content_rm_list = []
    if not os.path.exists(path_dictionary):  # 判断是否有路径文件夹
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        files = loadFiles(path_doc_root)  # 循环载入文件，返回类名(文件夹名)，文件内容
        for i, msg in enumerate(files):  # 使用enumerate遍历序号以及文件内容
            if i % n == 0:
                catg = msg[0]  # 获取类名
                content = msg[1]  # 获取内容
                content = contentSeg(content)  # 分词
                content = contentRm(content)  # 去掉停用词
                content_rm_list.append(content)
                if int(i / n) % 1000 == 0:  # 每处理1000个文档，打印
                    print('{t} *** {i} \t docs has been dealed'
                          .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

        dictionary = corpora.Dictionary(content_rm_list)  # 生成字典
        # 过滤词频较少的词语,减小矩阵
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5]
        dictionary.filter_tokens(small_freq_ids)
        # 消除id序列在删除词后产生的不连续的缺口
        dictionary.compactify()
        # 保存字典
        dictionary.save(path_dictionary)
        print('=== 词典已经生成 ===')
    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')

    """
    ===================================================================
            第二阶段，  开始将文档转化成tfidf
    """
    if not os.path.exists(path_tmp_tfidf):
        print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        # 如果指定的位置没有tfidf文档，则生成一个。如果有，则跳过该阶段
        if not dictionary:  # 如果跳过了第一阶段,则从指定位置读取词典
            dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        # 使用gensim的models，将文档转化成tfidf向量
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                content = msg[1]
                content = contentSeg(content)
                content = contentRm(content)
                content_bow = dictionary.doc2bow(content)  # 使用字典将内容转化成词袋
                content_tfidf = tfidf_model[content_bow]  # 将词袋向量幻化成tfidf向量
                tmp = corpus_tfidf.get(catg, [])  # dict.get 获取类别，如果false，返回列表[]
                tmp.append(content_tfidf)  # 添加新的tfidf向量到该类列表中
                if tmp.__len__() == 1:  # 如果这个类别长度为1，直接在定义字典中定义类别
                    corpus_tfidf[catg] = tmp

            if i % 10000 == 0:
                print('{i} files is dealed'.format(i=i))

        # 将tfidf中间结果存储起来
        catgs = list(corpus_tfidf.keys())  # 获取字典的key列表，以便分类序列化存储
        for catg in catgs:
            # 分类序列化存储tfidf向量
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg),
                                       corpus_tfidf.get(catg),
                                       id2word=dictionary
                                       )
            print('catg {c} has been transformed into tfidf vector'.format(c=catg))
        print('=== tfidf向量已经生成 ===')
    else:
        print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')

    """
     ===================================================================
             第三阶段，  开始将tfidf转化成lsi
    """
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
                t = file.split('.')[0]  # 文件存储类型xx.mm和xx.mm.index
                if t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus

        # 生成 lsi model
        os.makedirs(path_tmp_lsi)
        corpus_tfidf_total = []  # 所有tfidf向量列表
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp

        # 使用gensim models.LsiModel生成lsi向量
        # 模型说明: LSI,英文：Latent Semantic Indexing的缩写，中文意译是潜在语义索引
        # LSA(LSI)使用SVD来对单词-文档矩阵进行分解。SVD可以看作是从单词-文档矩阵中发现不相关的索引变量(因子)，
        # 将原来的数据映射到语义空间内。在单词-文档矩阵中不相似的两个文档，可能在语义空间内比较相似。
        # 参数说明
        # corpus: tfiidf向量列表
        # id2word: 字典
        # num_topics: 主题
        lsi_model = models.LsiModel(corpus=corpus_tfidf_total, id2word=dictionary, num_topics=50)

        # 将lsi模型存储到磁盘上
        lsi_file = open(path_tmp_lsimodel, 'wb')
        pkl.dump(lsi_model, lsi_file)
        lsi_file.close()
        del corpus_tfidf_total  # lsi model已经生成，释放变量空间
        print("--- lsi模型已经生成")

        # 生成corpus of lsi 并逐步去掉 corpus of tfidf
        corpus_lsi = {}
        for catg in catgs:
            corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lsi[catg] = corpu
            corpus_tfidf.pop(catg)  # 使用pop来删除，节省内存空间
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lsi, s=os.sep, c=catg),
                                       corpu,
                                       id2word=dictionary)
        print("=== lsi向量已经生成 ===")
    else:
        print('=== 检测到lsi向量已经生成，跳过该阶段 ===')

    """
    ===================================================================
            第四阶段，  分类
    """
    if not os.path.exists(path_tmp_predictor):
        print('=== 未检测到判断器存在，开始进行分类过程 ===')
        if not corpus_lsi:  # 如果跳过了第三阶段
            print('--- 未检测到lsi文档，开始从磁盘中读取 ---')
            files = os.listdir(path_tmp_lsi)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)
            # 从磁盘中读取corpus
            corpus_lsi = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi, s=os.sep, c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            print('--- lsi文档读取完毕，开始进行分类 ---')

        tag_list = []
        doc_num_list = []
        corpus_lsi_total = []
        catg_list = []
        files = os.listdir(path_tmp_lsi)
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)

        for count, catg in enumerate(catg_list):
            tmp = corpus_lsi[catg]  # 读取catg对应的lsi向量列表
            tag_list += [count] * tmp.__len__()  # 定义tag_list,lsi向量列表生成对应数量的标签
            doc_num_list.append(tmp.__len__())  # 列表对应的文档数量列表
            corpus_lsi_total += tmp  # lsi向量总列表
            corpus_lsi.pop(catg)  # pop该类别

        # 将gensim的mm表示转化成numpy矩阵表示
        data = []  # 数据
        rows = []  # 矩阵行
        cols = []  # 矩阵列
        line_count = 0
        for line in corpus_lsi_total:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1

        lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
        # 生成训练集和测试集
        rarray = np.random.random(size=line_count)  # 随机矩阵，用于区分训练集合
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i] < 0.8:  # 随机值<0.8 定义为训练集合 其余为测试机和
                train_set.append(lsi_matrix[i, :])
                train_tag.append(tag_list[i])
            else:
                test_set.append(lsi_matrix[i, :])
                test_tag.append(tag_list[i])

        # 生成分类器
        predictor = svmClassify(train_set, train_tag, test_set, test_tag)
        x = open(path_tmp_predictor, 'wb')
        pkl.dump(predictor, x)  # 保存分类器
        x.close()
    else:
        print("=== 检测到分类器已经生成，跳过该阶段 ===")

    """ 
     ===================================================================
            第五阶段，  对新文本进行判断
    """
    if not dictionary:
        dictionary = corpora.Dictionary.load(path_dictionary)
    if not lsi_model:
        lsi_file = open(path_tmp_lsimodel, 'rb')
        lsi_model = pkl.load(lsi_file)
        lsi_file.close()
    if not predictor:
        x = open(path_tmp_predictor, 'rb')
        predictor = pkl.load(x)
        x.close()
    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)

    demo_doc = """
    这次大选让两党的精英都摸不着头脑。以媒体专家的传统观点来看，要选总统首先要避免失言，避免说出一些“offensive”的话。希拉里，罗姆尼，都是按这个方法操作的。罗姆尼上次的47%言论是在一个私人场合被偷录下来的，不是他有意公开发表的。今年希拉里更是从来没有召开过新闻发布会。
    川普这种肆无忌惮的发言方式，在传统观点看来等于自杀。
    """
    print("原文本内容为:")
    print(demo_doc)
    demo_doc = list(jieba.cut(demo_doc, cut_all=False))  # 分词
    for elem in demo_doc:
        print elem
    demo_doc = contentRm(demo_doc)  # 删除停用词
    for elem in demo_doc:
        print elem
    demo_bow = dictionary.doc2bow(demo_doc)  # 词袋
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    demo_tfidf = tfidf_model[demo_bow]  # tfidf向量
    demo_lsi = lsi_model[demo_tfidf]  # lis向量
    data = []
    cols = []
    rows = []
    for item in demo_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    demo_matrix = csr_matrix((data, (rows, cols))).toarray()  # lsi向量矩阵
    x = predictor.predict(demo_matrix)
    print(x[0])
    print('分类结果为：{x}'.format(x=catg_list[x[0]]))


if __name__ == '__main__':
    main()
