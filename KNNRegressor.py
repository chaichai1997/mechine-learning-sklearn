# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, model_selection


"""
  在sin函数的基础上添加噪声点
"""


def create_regression_data(n):
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 1*(0.5-np.random.rand(int(n/5)))
    return model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


"""
K近邻回归法
def __init__(self, n_neighbors=5, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
                 参数：
        n_neighbors：一个整数，指定K值
        weights”一个字符串或可调用对象，指定投票权重类型
            uniform：本节点的所有邻居节点投票权重都相等
            distance: 本节点所有邻居节点的投票权权重与距离成反比
        algorithm：字符串，指定计算最近邻的算法
            ball_tree:使用BallTree算法
            kd_tree:使用KDtree算法
            brute：使用暴力搜索
            auto：自动
        leaf_size：整数，指定BallTree或者KDTree叶节点规模
        metric：指定距离度量，默认为minkowski距离
        p:指定在minkowski度量上的指数
        n_jobs:并行性，-1
    方法
        fit(X, Y)训练模型
        predict（X）预测
        score() 预测准确度                                                                 
        predict_proba(X)：返回样本为每种标记的概率
        kneighboes(X, n_neifhbors, return distance):返回样本的K近邻点   
"""


def test_KNNRegressor(*args):
    X_train, X_test, y_train, y_test = args
    regr = neighbors.KNeighborsRegressor()
    regr.fit(X_train, y_train)
    print("training score:", regr.score(X_train, y_train))
    print("Testing score:", regr.score(X_test, y_test))


"""
测试K值机投票策略对预测性能的影响
"""


def test_KNNRegressor_k_W(*args):
    X_train, X_test, y_train, y_test = args
    ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in weights:
        training_scores = []
        testing_scores = []
        for k in ks:
            regr = neighbors.KNeighborsRegressor(weights=i, n_neighbors=k)
            regr.fit(X_train, y_train)
            training_scores.append(regr.score(X_train, y_train))
            testing_scores.append(regr.score(X_test, y_test))
        ax.plot(ks, testing_scores, label="test score: weight=%s" % i)
        ax.plot(ks, training_scores, label="training score: weight=%s" % i)
    ax.plot(loc="best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNNRegressor")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = create_regression_data(1000)
    # test_KNNRegressor(X_train, X_test, y_train, y_test)
    test_KNNRegressor_k_W(X_train, X_test, y_train, y_test)
