# -*- coding: utf-8 -*-
import numpy as np
import  matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


"""
__init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components  # 指定降维后维数
        self.copy = copy       # 值为False，直接使用原始数据训练，训练结束后新数据覆盖原始数据
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
    属性：
        components_：主成分数组
        explained_variance_ratio_:一个数组，元素是每个主成分的explained variance的比列
        mean_:一个数组，元素是每个特征值的统计平均值
        n_components_:一个整数，指示主成分有多少个元素
    方法：
        fit():训练模型
        transform(X):执行降维
"""


def test_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    print("explained variance ratio :%s" %str(pca.explained_variance_ratio_))


"""
绘制降维后样本分布图
"""


def plot_PCA(*args):
    x, y = args
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    x_r = pca.transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0),
              (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0),
              (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for i,c in zip(np.unique(y), colors):
        position = y == i
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target= %d" %i,color=c)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()


"""
kernelPCA
    参数：
        n_components:一个整数，指定降维后维数
        kernel:一个字符串，指定核函数
            linear: 线性核
            poly： 多项式核
            rbf： 高斯核函数
            sigmoid： 
        degree： 一个整数。当核函数是多项式核函数是，指定多项式的系数
        alpha： 岭回归的超参数
    方法：
        inverse_transform(x)： 执行升维
"""


def test_KPCA(*args):
    x, y =args
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for k in kernels:
        kpca = decomposition.KernelPCA(n_components=None, kernel=k)
        kpca.fit(x, y)
        print('kernel=%s-->lambdas: %s' %(k, kpca.lambdas_))


if __name__ == '__main__':
    x, y = load_data()
    # test_PCA(x, y)
    # plot_PCA(x, y)
    test_KPCA(x, y)