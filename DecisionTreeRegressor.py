# -*- coding: utf-8 -*-
"""
回归决策树：DecisonTreeRegressor,sklearn中采用CART决策树算法
    该算法应用于回归问题
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt


"""
随机产生一个数据集：
    在sin(X)的基础上添加了若干个随机噪声。
    X在0-1间随机产生。
    每隔五个点添加一个随机噪声
参数：n为数据集容量
"""


def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    Y = np.sin(X).ravel()
    noise_num = int(n / 5)
    Y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return model_selection.train_test_split(X, Y, test_size=0.25, random_state=1)


"""
__init__(self,
                 criterion="mse", 指定切分质量的评价准则，默认表示均方误差
                 splitter="best", 指定切分原则，best最优切分，random随即切分
                 max_depth=None,  树的最大深度
                 min_samples_split=2, 内部节点包含的最少的样本数
                 min_samples_leaf=1, 叶子节点包含的最少的样本数
                 min_weight_fraction_leaf=0., 叶节点中样本最小权重系数
                 max_features=None, 
                 random_state=None,
                 max_leaf_nodes=None,  指定叶节点的最大数量
                 min_impurity_decrease=0., 
                 min_impurity_split=None,
                 presort=False):
属性：
    feature_importances_:特征的重要程度
    max_features_:max_features的推断值
    n_features_：fit后的特征数量
    n_outputs_：执行fit之后，输出的数量
    tree_:一个Tree对象，即底层决策树
方法：
    fit()
    predict() 
    score()
scatter:绘制散点图，可指定点的大小与颜色
结果：
    使用默认最优切分，训练集拟合效果好，出现一定程度过拟合
"""


def test_DecisionTreeRegressor(*args):
    X_train, X_test, y_train, y_test = args
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)
    print("Training score: %f" % (regr.score(X_train, y_train)))
    print("Testing score: %f" % regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train, y_train, label="train sample", c='g')
    ax.scatter(X_test, y_test, label="test sample", c='r')
    ax.plot(X, Y, label="predict_value", linewidth=2, alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


"""
测试随机划分与最优化分对模型性能的影响
"""


def test_DecisionTreeRegressor_splitter(*args):
    X_train, X_test, y_train, y_test = args
    splitters = ['best', 'random']
    for i in splitters:
        regr = DecisionTreeRegressor(splitter=i)
        regr.fit(X_train, y_train)
        print('splitter:'+i)
        print("Training score: %f" % (regr.score(X_train, y_train)))
        print("Testing score: %f" % regr.score(X_test, y_test))


"""
测试决策树深度对模型效率的影响
决策树越深，模型越复杂
"""


def test_DecisiontreeRegressor_depth(*args, maxdepth):
    X_train, X_test, y_train, y_test = args
    depths = np.arange(1, maxdepth)
    training_score = []
    test_score = []
    for i in depths:
        regr = DecisionTreeRegressor(max_depth=i)
        regr.fit(X_train, y_train)
        training_score.append(regr.score(X_train, y_train))
        test_score.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_score, label="training score", color= 'r')
    ax.plot(depths, test_score, label="test score", color='b')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = creat_data(100)
    # test_DecisionTreeRegressor(X_train, X_test, y_train, y_test)
    # test_DecisionTreeRegressor_splitter(X_train, X_test, y_train, y_test)
    test_DecisiontreeRegressor_depth(X_train, X_test, y_train, y_test, maxdepth=20)



