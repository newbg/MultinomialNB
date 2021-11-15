import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from nbc import MultinomialNB # 导入自定义的多项式朴素贝叶斯

# 获取语义较弱的停顿词
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

# 从文件路径得到训练集和对应的类别
def getTraindata(data_path):
    train_data = []
    train_class = []
    path_list = os.listdir(data_path)
    for path in path_list:
        tmp_path = data_path + "/" + path

        new_path_list = os.listdir(tmp_path)
        for new_path in new_path_list:
            file_path = tmp_path + "/" + new_path

            with open(file_path, mode="r", encoding='utf-8') as fp: #ANSI
                raw_data = fp.read()
                data = list(jieba.cut(raw_data, cut_all=False))
                train_data.append(data)
                train_class.append(path)

    return train_data, train_class


# 从文件路径得到测试集
def getTestdata(data_path):
    test_data = []
    path_list = os.listdir(data_path)
    for path in path_list:
        file_path = data_path + "/" + path
        with open(file_path, mode="r", encoding='ANSI') as fp: #ANSI
            raw_data = fp.read()
            data = list(jieba.cut(raw_data, cut_all=False))
            test_data.append(data)
    return test_data

# 获取字典集
def getDic(data_path):
    data = []
    file = open(data_path, "r")
    for line in file.readlines():
        curline = line.strip().split(" ") # 去除每一行字符串首尾的空白字符
        data.append(curline[0])
    return data

# 特征选取
def word_select(dic, stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(0, len(dic), 3):
        # n表示选取特征的维度，一共大约21万维，每隔三个选取一个特征，降到六万维
        if n > 60000:
            break
        # 不选词典中数字、停顿词作为特征
        if not dic[t].isdigit() and dic[t] not in stopwords_set and 1<len(dic[t])<5:
            feature_words.append(dic[t])
            n += 1
    return feature_words


# 提取文本特征向量
def getTextfeature(train_data, test_data, feature_words):
    # 把一个txt文件转换为特征向量
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_features = [text_features(text, feature_words) for text in train_data]
    test_features = [text_features(text, feature_words) for text in test_data]
    return train_features,test_features


if __name__ == '__main__':
    print("begin!")

    # 训练集 测试集路径
    # train_data_path = "./Training Dataset" # 全部训练集 9*1989
    test_data_path = "./Test Dataset"  #老师给定的测试集

    train_data_path = "./Training"  # 自定义的训练数据局 9*10篇新闻
    # test_data_path = "./Test"  #自定义的测试集

    # 词典文件路径
    dic_path = "./SegDict.TXT"

    # 导入词典集
    dic = getDic(dic_path)
    # print(dic[0], len(dic[0]))

    # 获取训练集 测试集
    train_data, train_class = getTraindata(train_data_path)

    test_data = getTestdata(test_data_path)

    # 获取停顿字符集合
    stopwords_path = './stopwords_cn.txt'
    stopwords = MakeWordsSet(stopwords_path)
    # print(type(stopwords))

    # 特征选择 从词典集合中选取语义较强的作为特征
    feature_words = word_select(dic, stopwords)


    # 获取训练集和测试集的特征向量
    train_features, test_features = getTextfeature(train_data, test_data, feature_words)

    # 把分类标签的最后两位作为类名 如C000024->24
    train_class = list(train_class)
    for index, item in enumerate(train_class):
        train_class[index] = int(item[-2:])

    # 模型定义
    classifier = MultinomialNB(alpha=1.0, fit_prior=True)

    # 参数类型转换为矩阵
    train_feature_list = np.array(train_features)
    train_class_list = np.array(train_class)
    test_feature_list = np.array(test_features)

    # 模型训练
    classifier.fit(train_feature_list, train_class)

    # 输出预测结果
    print(classifier.predict(test_feature_list))

    print("end!")