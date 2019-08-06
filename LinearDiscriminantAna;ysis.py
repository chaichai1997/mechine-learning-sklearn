# -*- coding: utf-8 -*-
"""
LDA:线性判别分析，使样例投影到一条直线，同类样例的投影点尽可能接近，不同类距离尽可能远
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, model_selection


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    for i in range(0, 10):
        print(iris.data[i], iris.target[i])
    return model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                            random_state=0, stratify=y_train)


"""
线性判别分析
_init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
参数：
    solver:指定求解最优化的方法
    shrinkage：常在训练样本小于特征数量的场合下使用。仅在solver='lsqr'(最小平方差算法)或eigen下才有意义
    priors：数组，依次制定类别的先验概率
    n_components：整数，数据降维后的维度
    store_covariance：布尔值，是否计算协方差矩阵
    tol：指定SVD收敛阈值
属性：
    coef_: 权重向量
    intercept_:b值
    covariance:数组，依次给出每个类别的协方差矩阵
    means_：数组，给出每个类别的均值向量
    xbar_:给出整体样本的均值向量
    n_iter_:实际迭代次数
方法：
    fit(x, y):训练模型
    predict(X):用模型预测并返回预测值
    predict_log_proba(X):返回一个数组，数组的元素依次是X预测为各个类别概率的对数值
    predict_proba(X):返回一个数组，数组的元素依次是X预测为各个类别概率值
    score(x, y)：模型预测准确度
    
    
    
"""


def test_LinearDiscriminantAnalysis(*args):
    X_train, X_test, y_train, y_test = args
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print('Coefficients:%s , intercept %s' % (lda.coef_, lda.intercept_))
    print('Score: %.2f' % lda.score(X_test, y_test))


def plot_LDA(convert_X, y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    color = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([0, 1, 2],color, markers):
        pos = (y == target).ravel()
        X = convert_X[pos,:]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker,
                   label= "label %d" %target)
    ax.legend(loc="best")
    fig.suptitle("Iris After LDA")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    X = np.vstack((X_train, X_test))
    Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X, Y)
    converted_X = np.dot(X, np.transpose(lda.coef_))+lda.intercept_
    plot_LDA(converted_X, Y)