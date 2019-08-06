# -*- coding: utf-8 -*-
from sklearn import datasets, model_selection, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    digit = datasets.load_digits()
    return model_selection.train_test_split(digit.data, digit.target,
                                            test_size=0.25, random_state=0)

"""
高斯贝叶斯分类器
__init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
    属性：
    class_prior_:一个数组，形状为(n_classes,)，是每个类别的概率
    class_count_:一个数组，形状为(n_classes,),是每个类别包含的训练样本数量
    theta_:一个数组，形状为(n_classes,),是每个类别上每个特征的均值
    sigma_:一个数组，形状为(n_classes,)，是每个类别上的每个特征的标准差
    方法：
    fit()：训练模型
    partial_fit():追加训练模型
    predict（）：预测
    predict_log_proba():返回一个数组，数组元素依次为各个类别的概率值
    score():预测准确度
"""


def test_GaussianNB(*args):
    X_train, X_test, y_train, y_test = args
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train, y_train)
    print('training score:', cls.score(X_train, y_train))
    print('testing score:', cls.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_GaussianNB(X_train, X_test, y_train, y_test)

