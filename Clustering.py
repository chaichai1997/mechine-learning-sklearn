# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import mixture


def create_data(centers, num=100, std=0.7):
    x, label_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return x, label_true


def plot_data(*args):
    x, label_true = args
    labels = np.unique(label_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    color = 'rgbyckm'
    for i,l in enumerate(labels):
        position = label_true == l
        ax.scatter(x[position, 0], x[position, 1],
                   label="cluster %d" % l,
                   color=color[i%len(color)])
    ax.legend(loc="best", framealpha=0.5)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[1]")
    ax.set_title("data")
    plt.show()


"""
__init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):
    参数：
    n_clusters: 一个整数，指定分类簇的数量
    init：字符串，初始化均值向量的策略
    n_init:整数，制定了k均值算法运行次数
    max_iter：整数，指定了单轮k均值算法中，最大的迭代次数
    precompute_distances：是否提前计算样本间距离
    tol：一个浮点数，指定算法收敛的阈值
    属性：
    cluster_centers_:给出分类簇的均值向量
    labels_:给出每个样本所属簇标记
    intertia_:给出每个样本距离他们各自最近的簇中心的距离之和
    方法：
    fit():训练模型
    predict(x)：预测样本所属簇
    fit_predcit(x,y):训练墨香并预测样本所属簇
    
"""


def test_kmeans(*args):
    x, label_true = args
    clst = cluster.KMeans()
    clst.fit(x)
    predicted_label = clst.predict(x)
    print("ARI:%s" % adjusted_mutual_info_score(label_true, predicted_label))
    print("Sum center distance %s" % clst.inertia_)


"""
 测试簇的数量对性能的影响
"""


def test_kmeans_nclusters(*args):
    x, labels_true = args
    nums = range(1, 50)
    ARIs = []
    Distance = []
    for i in nums:
        clst = cluster.KMeans(n_clusters=i)
        clst.fit(x)
        predicted_label = clst.predict(x)
        ARIs.append(adjusted_mutual_info_score(labels_true, predicted_label))
        Distance.append(clst.inertia_)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distance, marker="o")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("Kmeans")
    plt.show()


"""
密度聚类
__init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
    参数：
        eps： 用于确定邻域大小
        min_samples: Minpts参数,用于确定核心对象
        metric:一个字符串或可调用对象，用于计算距离
        algorithm:一个字符串，用于计算两点间距离并找出最近邻的点
    属性：
        core_sample_indices_:核心样本在原始训练集中的位置
        components_：核心样本的一份副本
        labels_：每个样本所属簇标记                                  
    方法：
        fit
        fit_predict(x)
"""


def test_DBSCAN(*args):
    x, label_true = args
    clst = cluster.DBSCAN()
    predict_label = clst.fit_predict(x)
    print("ARI: %s" % adjusted_mutual_info_score(label_true, predict_label))
    print("Core asmple num: %d" % len(clst.core_sample_indices_))


if __name__ == '__main__':
    x, label_true = create_data([[1, 1], [2, 2], [1, 2], [10, 20]],
                1000, 0.5)
    # plot_data(x, label_true)
    # test_kmeans(x, label_true)
    # test_kmeans_nclusters(x, label_true)
    test_DBSCAN(x, label_true)