# -*- coding: utf-8 -*-
"""
SVM:
    优点：本质上是非线性方法，容易抓住数据与特征之间的非线性关系，可以避免神经网络结构选择和局部极小点问题
    缺点：对缺失数据敏感，对非线性问题没有通用解决方案
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection, svm


"""
SVM回归问题，使用SKlearn自带糖尿病病人数据集
"""


def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                             test_size=0.25, random_state=0)


"""
SVM分类问题，使用鸢尾花数据集
"""


def load_data_classfication():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                            random_state=0, stratify=y_train)


"""
线性SVC:LinearSVM,可以用于二分类，也可用于多分类
LinearSVM:
     __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
     penalty：惩罚的范数，指定为'l1'或'l2'
     C:浮点数，惩罚参数
     loss：字符串，损失函数
     dual：布尔值，True则解决对偶问题，False则解决原始问题
     tol：迭代终止的阈值
     multi_class：字符串，指定多累分类问题的策略
     fit_intercept：布尔值，True计算截距
     class_weight：各个类的权重
     max_iter：最大迭代次数
  属性：
  intercept_：数组，给出截距
  coef_：数组，给出各个特征的权重
"""


def test_LinearSVC(*args):
    X_train, X_test, y_train, y_test = args
    cls = svm.LinearSVC()
    cls.fit(X_train, y_train)
    print("cofficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
    print("score: %.2f" % cls.score(X_test, y_test))


"""
 验证损失函数对线性支持向量机的预测性能影响
"""


def test_LinearSVM_Loss(*args):
    X_train, X_test, y_train, y_test = args
    losses = ['hinge', 'squared_hinge']
    for i in losses:
        cls = svm.LinearSVC(loss=i)
        cls.fit(X_train, y_train)
        print("loss:", i)
        print("cofficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
        print("score: %.2f" % cls.score(X_test, y_test))


"""
 验证惩罚项对线性支持向量机的预测性能影响
"""


def test_LinearSVM_L12(*args):
    X_train, X_test, y_train, y_test = args
    l = ['l1', 'l2']
    for i in l:
        cls = svm.LinearSVC(penalty=i, dual=False)
        cls.fit(X_train, y_train)
        print("loss:", i)
        print("cofficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
        print("score: %.2f" % cls.score(X_test, y_test))


"""
 验证惩罚项系数对线性支持向量机的预测性能影响
"""


def test_LinearSVM_C(*args):
    X_train, X_test, y_train, y_test = args
    C = np.logspace(-2, 1)
    train_score = []
    test_score = []
    for i in C:
        cls = svm.LinearSVC(C=i)
        cls.fit(X_train, y_train)
        train_score.append(cls.score(X_train, y_train))
        test_score.append(cls.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(C, train_score, label="training score")
    ax.plot(C, test_score, label="testing score")
    ax.set_xlabel('c')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_title("LinearSVC")
    ax.legend(loc='best')
    plt.show()


"""
非线性SVM分类 SVC,svc训练的时间复杂度是采样点数量的平方。
def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
     C:浮点数，惩罚参数
     kernel：一个字符串，指定核函数linear线性核；poly，多项式核；rbf，高斯核函数；sigmoid
     degree:一个整数。当核函数是多项式核函数时，指定多项式的系数
     coef0：指定核函数中的自由项
     tol：迭代终止的阈值
     shrinking：布尔值，是否采用启发式收缩
     class_weight：各个类的权重
     max_iter：最大迭代次数
  属性：
  support：一个数组，支持向量的下标
  n_support_：每一个支持向量的个数
  dual_coef_:一个数组，在分类决策函数中每个支持向量的系数
  coef_：数组，给出各个特征的权重，只有在线性核中有效
    
"""

"""
    线性核
"""


def test_SVC_linear(*args):

    X_train, X_test, y_train, y_test = args
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train, y_train)
    print("cofficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
    print("score: %.2f" % cls.score(X_test, y_test))


"""
多项式核
"""
def test_SVC_poly(*args):
    X_train, X_test, y_train, y_test = args
    fig = plt.figure()
    degrees = range(1, 20)
    train_score = []
    test_score = []
    for i in degrees:
        cls = svm.SVC(kernel='poly', degree=i)
        cls.fit(X_train, y_train)
        train_score.append(cls.score(X_train, y_train))
        test_score.append(cls.score(X_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(degrees, train_score, label="training score", marker='*')
    ax.plot(degrees, test_score, label="testing score", marker='^')
    ax.set_title("SVC_poly_degree")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="beat", framealpha=0.5)
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data_classfication()
    # test_LinearSVC(X_train, X_test, y_train, y_test)
    # test_LinearSVM_Loss(X_train, X_test, y_train, y_test)
    # test_LinearSVM_L12(X_train, X_test, y_train, y_test)
    # test_LinearSVM_C(X_train, X_test, y_train, y_test)
    # test_SVC_linear(X_train, X_test, y_train, y_test)
    test_SVC_poly(X_train, X_test, y_train, y_test )