# -*- coding: utf-8 -*-
import  matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, ensemble


""" 
    回归问题，加载糖尿病病人数据集
"""


def load_dataset_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                            test_size=0.25, random_state=0)


""" 
    分类问题，加载糖尿病病人数据集
"""


def load_dataset_classfication():
    diabetes = datasets.load_digits()
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                            test_size=0.25, random_state=0)


"""
    adaBoostClassfier分类器
     def __init__(self,
                 base_estimator=None,   # 基础分类器对象
                 n_estimators=50,        # 基分类器数量
                 learning_rate=1.,        # 浮点数，用于减少每一步的补偿
                 algorithm='SAMME.R',    # 字符串，指定算法 SAMME与SAMME.R两种
                 random_state=None):
     属性：
        estimators_:所有训练过的基础分类器
        classes_:所有的类别标签
        n_classes_:类别数量
        estimators_weights_：每个基础分类器的权重
        estimators_errors_:每个基础分类器的分类误差
        feature_importance_：每个特征的重要性
    方法：
        fit(X,y,sample_w):训练模型
        predict():预测模型
        score(X,y,[sample]):预测准确率
    
"""


def test_AdaBoostClassifier(*args):
    X_train, X_test, y_train,  y_test = args
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    estimators_num = len(clf.estimator_errors_)
    X = range(1, estimators_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traning score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaboostClassfier")
    plt.show()


"""
不同类型的个体分类器的影响
"""


def test_AdaBoostClassfier_base_classfier(*args):
    from sklearn.naive_bayes import GaussianNB
    X_train, X_test, y_train,  y_test = args
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    """默认个体分类器 """
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train, y_train)
    estimators_num = len(clf.estimator_errors_)
    X = range(1, estimators_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traning score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoost with Decision Tree")

    """贝叶斯分类器 """
    ax = fig.add_subplot(2, 1, 2)
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1, base_estimator=GaussianNB())
    clf.fit(X_train, y_train)
    estimators_num = len(clf.estimator_errors_)
    X = range(1, estimators_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traning score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoost with Gaussian Bayes")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train,  y_test = load_dataset_classfication()
    # test_AdaBoostClassifier(X_train, X_test, y_train,  y_test)
    test_AdaBoostClassfier_base_classfier(X_train, X_test, y_train,  y_test)