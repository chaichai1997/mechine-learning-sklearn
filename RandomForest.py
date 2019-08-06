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
随机森林分类
__init__(self,
                 n_estimators='warn',  # 随机森林中决策树的数量
                 criterion="gini",     # 单个决策树的criterion参数
                 max_depth=None,       # 单个决策树的深度，决策树最大深度增高，每个树的性能也提高；
                                                        决策树最大深度提高，决策树多样性也在增大
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
"""


def test_RandomForesrClassfier(*args):
    X_train, X_test, y_train, y_test = args
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, y_train)
    print("training score:%f" %clf.score(X_train, y_train))
    print("testing score: %f" %clf.score(X_test, y_test))


def test_RandomForestRegressor(*args):
    X_train, X_test, y_train, y_test = args
    regr = ensemble.RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("training score:%f" %regr.score(X_train, y_train))
    print("testing score: %f" %regr.score(X_test, y_test))


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = load_dataset_classfication()
    # test_RandomForesrClassfier(X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = load_dataset_regression()
    test_RandomForestRegressor(X_train, X_test, y_train, y_test)