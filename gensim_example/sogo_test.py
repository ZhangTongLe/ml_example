#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: liujm
@contact: kaka206@163.com
@software: PyCharm
@file: sogo_test.py
@time: 2017/9/13 14:18
"""

# 加载库
import os
import jieba
import logging
import codecs
from gensim import corpora, models, similarities

# 日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

"""
使用codecs 处理编码问题
"""


def savefile(savepath, content):
    """
    :Descirption : 保存文件
    :param savepath: 保存路径
    :param content: 需要保存的内容
    :return:
    """
    fp = codecs.open(savepath, 'w+', 'gb18030')
    fp.write(content)
    fp.close()


def readfile(path):
    """
    Descriprtion: 读取文件
    :param path:
    :return:
    """
    fp = codecs.open(path, "r", 'gb18030', errors='ignore')
    content = fp.read()
    fp.close()
    return content


def corpus_segment(corpus_path):
    """
    Description: 读取语料库后进行分词
    :param corpus_path: 读取的语料库路径
    :return:
    """
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录

    '''
    其中子目录的名字就是类别名，例如： 
    train_corpus/art/21.txt中，'train_corpus/'是corpus_path，'art'是catelist中的一个成员 
    '''
    content_seg_list = []
    label_dict = {}
    # 获取每个目录（类别）下所有的文件
    count = 0
    for mydir in catelist:
        '''
        这里mydir就是train_corpus/art/21.txt中的art（即catelist中的一个类别） 
        '''
        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径如：train_corpus/art/

        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本

        ''''' 
        train_corpus/art/中的 
        21.txt, 
        22.txt, 
        23.txt 
        ... 
        file_list=['21.txt','22.txt',...] 
        '''
        for file_path in file_list:  # 遍历类别目录下的所有文件
            fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
            content = readfile(fullname)  # 读取文件内容
            content = content.replace("\r\n", "")  # 删除换行
            content = content.replace(" ", "")  # 删除空行、多余的空格
            content_seg = jieba.cut(content)  # 为文件内容分词
            content_seg_list.append(content_seg)  #
            label_name = mydir + "_" + file_path.replace(".txt", "")
            label_dict[count] = label_name
            count += 1

    print "中文语料分词结束！！！"
    return content_seg_list, label_dict


def corpus_stop(content_seg_list, stop_list):
    """
    Description: 去掉常用停用词
    :param content_seg_list: 分词后的语料库列表
    :param stop_list: 常用停用词列表
    :return:
    """
    after_stop_content_list = []
    for content in content_seg_list:
        after_stop_content = []
        for words in content:
            if words not in stop_list:
                after_stop_content.append(words)

        after_stop_content_list.append(after_stop_content)

    print "去除停词结束！！！"
    return after_stop_content_list


def get_dictionary(after_stop_content_list):
    """
    Description: 获取字典
    :param after_stop_content_list: 去掉停用词后的语料列表
    :return:
    """
    return corpora.Dictionary(after_stop_content_list)


def get_lsi_model(after_stop_content_list, dictionary, topics_num=9):
    """
    Description: 获取lsi 模型,相似性矩阵
    :param topics_num: 模型分解后的主体数量
    :param after_stop_content_list: 去掉停用词后的语料列表
    :param dictionary: 获取字典
    :return: lsi:lsi模型  index :相关性矩阵
    """
    corpus = [dictionary.doc2bow(text) for text in after_stop_content_list]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topics_num)

    index = similarities.MatrixSimilarity(lsi[corpus])

    return lsi, index


def lsi_train(corpus_path, stop_path, model_save_path, dictionary_save_path, index_save_path):
    """
    Description: lsi模型训练程序
    :param index_save_path:
    :param dictionary_save_path: 字典保存路径
    :param corpus_path: 语料库存放路径
    :param stop_path: 停用词存放路径
    :param model_save_path: 模型保存路径
    :return:
    """
    # 分词
    content_seg_list, label_dict = corpus_segment(corpus_path)
    stop_path = stop_path
    stop_list = set(readfile(stop_path).split("\r\n"))
    after_stop_content_list = corpus_stop(content_seg_list, stop_list)
    for words in after_stop_content_list[0]:
        print words

    dictionary = get_dictionary(after_stop_content_list)
    lsi, index = get_lsi_model(after_stop_content_list, dictionary)

    lsi.save(model_save_path)
    dictionary.save(dictionary_save_path)
    index.save(index_save_path)


def lsi_test(model_path, dictionary_path, index_path, temp_list):
    lsi = models.LsiModel.load(model_path)
    dictionary = corpora.Dictionary.load(dictionary_path)
    index = similarities.MatrixSimilarity.load(index_path)

    temp_bow = dictionary.doc2bow(temp_list)
    temp_lsi = lsi[temp_bow]
    print ' '.join(temp_list)
    sims = index[temp_lsi]
    print temp_lsi
    label, score = sorted(temp_lsi, key=lambda item: abs(-item[1]))[0]
    print label
    print len(sims)
    # sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # for k, sim in sort_sims:
    #     print k, ':', sim


def main():
    corpus_path = "D:/work/Reduced/"
    stop_path = "D:/work/stopwords.txt"
    model_save_path = "D:/Users/lenovo/PycharmProjects/ml_example/gensim_example/lsi_model"
    dictionary_save_path = "D:/Users/lenovo/PycharmProjects/ml_example/gensim_example/lsi_dic"
    index_save_path = "D:/Users/lenovo/PycharmProjects/ml_example/gensim_example/lsi_index"

    if not os.path.exists(model_save_path):
        lsi_train(corpus_path=corpus_path, stop_path=stop_path, model_save_path=model_save_path,
                  dictionary_save_path=dictionary_save_path, index_save_path=index_save_path)

    # lsi_test(model_path=model_save_path, dictionary_path=dictionary_save_path,
    #          index_path=index_save_path, temp_list=['NBA', '篮球', '骑士队'])

    content_seg_list, label_dict = corpus_segment(corpus_path)
    stop_path = stop_path
    stop_list = set(readfile(stop_path).split("\r\n"))
    after_stop_content_list = corpus_stop(content_seg_list, stop_list)
    for words in after_stop_content_list[0]:
        print words

    lsi = models.LsiModel.load(model_save_path)
    dictionary = corpora.Dictionary.load(dictionary_save_path)
    # index = similarities.MatrixSimilarity.load(index_save_path)

    out_dict = {}

    for rank, file_list in enumerate(after_stop_content_list):
        file_bow = dictionary.doc2bow(file_list)
        file_lsi = lsi[file_bow]
        predict, score = sorted(file_lsi, key=lambda item: abs(-item[1]))[0]
        # print predict, label_dict[rank]
        out_dict[predict] += 1

    # for label, predict_value in all_dict.iteritems():
    #     print label, predict_value
    for predict, value in out_dict.iteritems():
        print predict, value


if __name__ == "__main__":
    main()
