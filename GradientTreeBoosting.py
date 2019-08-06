# -*- coding: utf-8 -*-
import sklearn
import numpy as np
from sklearn import datasets, model_selection, ensemble
from matplotlib import pyplot as plt
"""
基于梯度提升树的分类与回归问题
"""


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
梯度提升决策树用作分类
def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):
            loss：损失函数 deviance默认 exponential指数损失函数,
            learning_rate 学习率
            n_estimators整数，指定基础决策树数量
            subsample=1.0浮点数，训练样本子集占原始训练集的大小
            warm_start=False,是否使用上一次的训练结果
        方法：
            stage_predict():返回一个数组，数组元素为每一轮迭代结束时集成分类器的预测值。
"""


def test_GradientBoostingClassifier(*args):
    X_train, X_test, y_train, y_test = args
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    print("training score:%f" %clf.score(X_train, y_train))
    print("testing score: %f" %clf.score(X_test, y_test))


def test_GradientBoostingClassifier_maxdepth(*args):
    from sklearn.naive_bayes import GaussianNB
    X_train, X_test, y_train,  y_test = args
    max_depth = np.arange(1, 20)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    test_score = []
    train_score = []
    for i in max_depth:
        clf = ensemble.GradientBoostingClassifier(max_depth=i, max_leaf_nodes=None)
        clf.fit(X_train, y_train)
        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))
    ax.plot(max_depth, train_score, label="Traning score")
    ax.plot(max_depth, test_score, label="Testing score")
    ax.set_xlabel("max depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gradient Boosting Classifier")
    plt.show()


"""
梯度提升树回归
__init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):
                 loss损失函数，ls平方损失函数， lad绝对值损失函数，huber两者结合
                 n_estimators整数，基础决策树个数,
                 max_depth:个体回归树最大深度
                 subsample：浮点数，单个回归数训练集占原始数据集的大小，小于1.0则为随机梯度提升回归数
                 max_feature:指定基础决策树的max_feature模型
"""


def test_GradientBoostingRegressor(*args):
    X_train, X_test, y_train, y_test = args
    regr = ensemble.GradientBoostingRegressor()
    regr.fit(X_train, y_train)
    print("training score:%f" % regr.score(X_train, y_train))
    print("testing score: %f" % regr.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset_classfication()
    X_train, X_test, y_train, y_test = load_dataset_regression()
    # test_GradientBoostingClassifier(X_train, X_test, y_train, y_test)
    # test_GradientBoostingClassifier_maxdepth(X_train, X_test, y_train, y_test)
    test_GradientBoostingRegressor(X_train, X_test, y_train, y_test)