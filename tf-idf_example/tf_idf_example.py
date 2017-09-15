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
import logging
import os
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def func():
    pass


class Main():
    def __init__(self):
        # 日志
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

        # 设置当前工作路径
        os.chdir("D:/work/liujm/2017/9/20170901/sklearn_exercise")
        """
        加载数据
        load_files:会自动讲文件夹名作为每一类的类名
        """
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

        # 使用贝叶斯进行训练
        clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

        # 对新的样本进行预测
        docs_new = ['God is love', 'OpenGL on the GPU is fast']
        X_new_counts = count_vect.transform(docs_new)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        predicted = clf.predict(X_new_tfidf)
        for doc, category in zip(docs_new, predicted):
            print "%r => %s" % (doc, twenty_train.target_names[category])

        # 建立使用管道来进行数据挖掘
        text_clf = Pipeline([
            ('vect', CountVectorizer(stop_words="english", decode_error="ignore")),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
        text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

        # 测试集分类准确率
        doc_test = twenty_test.data
        predicted = text_clf.predict(doc_test)
        print "bayes", np.mean(predicted == twenty_test.target)

        # 使用线性核支持向量机
        text_clf_2 = Pipeline([
            ('vect', CountVectorizer(stop_words="english", decode_error="ignore")),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, n_iter=5, random_state=42))
        ])
        _ = text_clf_2.fit(twenty_train.data, twenty_train.target)
        predicted = text_clf_2.predict(doc_test)

        print "svm:", np.mean(predicted == twenty_test.target)
        print metrics.classification_report(twenty_test, predicted,
                                            target_names=twenty_test.target_names)

        # 尝试使用超参数,选择参数
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
        }
        # 设置`n_jobs = -1`，计算机就会帮你自动检测并用上你所有的核进行并行计算。
        gs_clf = GridSearchCV(text_clf_2, parameters, n_jobs=2)
        #


if __name__ == '__main__':
    Main()
