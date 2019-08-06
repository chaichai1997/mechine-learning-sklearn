# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from sklearn import datasets, model_selection, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    digit = datasets.load_digits()
    return model_selection.train_test_split(digit.data, digit.target,
                                            test_size=0.25, random_state=0)


"""
多项式贝叶斯分类器
def __init__(self, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
    参数：
    alpha：浮点数，指定一个α值
    binarize:浮点数或者None，None则假定原始数据已经二值化
    fit_prior：布尔值，为True则不去学习P(y=ck)，以均匀分布替代
    class_prior:一个数组，指定了每个分类的先验概率
    属性：
    class_prior_:一个数组，形状为(n_classes,)，是每个类别的概率
    class_count_:一个数组，形状为(n_classes,),是每个类别包含的训练样本数量
    theta_:一个数组，形状为(n_classes,),是每个类别上每个特征的均值
    feature_log_prob_：一个数组独享，给出经验概率分布的对数值
    feature_count_：一个数组，是训练过程中，每个类别特征遇到的样本数
    方法：
    fit()：训练模型
    partial_fit():追加训练模型
    predict（）：预测
    predict_log_proba():返回一个数组，数组元素依次为各个类别的概率值的对数值
    predict_proba():返回一个数组，数组元素依次为各个类别的概率值的对数值
    score():预测准确度
"""


def test_BernoulliNB(*args):
    X_train, X_test, y_train, y_test = args
    cls = naive_bayes.BernoulliNB()
    cls.fit(X_train, y_train)
    print('training score:', cls.score(X_train, y_train))
    print('testing score:', cls.score(X_test, y_test))


"""
测试α对多项式贝叶斯分类器的预测性能的影响
结果：α大于100后，预测准确度受精度影响较大
"""


def test_BernoulliNB_alpha(*args):
    X_train, X_test, y_train, y_test = args
    alpha = np.logspace(-2, 5, num=200)
    train_score = []
    test_score = []
    for i in alpha:
        cls = naive_bayes.BernoulliNB(alpha=i)
        cls.fit(X_train, y_train)
        train_score.append(cls.score(X_train, y_train))
        test_score.append(cls.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha, train_score, label="Training score", c='r')
    ax.plot(alpha, test_score, label="Testing score", c='b')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("BernoulliNB")
    ax.set_xscale("log")
    ax.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_BernoulliNB(X_train, X_test, y_train, y_test)
    test_BernoulliNB_alpha(X_train, X_test, y_train, y_test)