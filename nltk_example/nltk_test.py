#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Liujm
@site:https://github.com/liujm7
@contact: kaka206@163.com
@software: PyCharm
@file: nltk_test.py
@time: 2017/9/14 
"""

from nltk.corpus import gutenberg
from nltk import FreqDist
from nltk import ConditionalFreqDist
from random import choice
import matplotlib.pyplot as plt


def show():
    print gutenberg.fileids()
    # 频率分布实例化
    fd = FreqDist()
    for word in gutenberg.words('austen-persuasion.txt'):
        fd[word] += 1

    print fd.N()
    print fd.B()
    # 得到前10个按频率排序后的词
    for word, value in sorted(fd.items(), key=lambda item: -item[1])[:10]:
        print word, value


def zipf_law_test():
    fd = FreqDist()
    for text in gutenberg.fileids():
        for word in gutenberg.words(text):
            fd[word] += 1
    ranks = []
    freqs = []

    for rank, (word, freq) in enumerate(sorted(fd.items(), key=lambda item: -item[1])):
        ranks.append(rank)
        freqs.append(freq)

    plt.loglog(ranks, freqs)
    plt.xlabel('frequency(f)', fontsize=14, fontweight='bold')
    plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.show()


def predict_next_word():
    cfd = ConditionalFreqDist()
    # 对于每个实例，统计给定词的下一个词数量
    prev_word = None
    for word in gutenberg.words('austen-persuasion.txt'):
        cfd[prev_word].update(word)
        prev_word = word

    word = 'therefore'
    i = 1
    while i < 20:
        print word
        lwords = cfd[word].samples()
        follower = choice(lwords)
        word = follower
        i += 1


class Main():
    def __init__(self):
        colors = ['red', 'green', 'red', 'blue', 'green', 'red']
        d = {}
        for color in colors:
            d[color] = d.get(color, 0) + 1
        for k, v in d.iteritems():
            print k, '===>', v


if __name__ == '__main__':
    Main()
