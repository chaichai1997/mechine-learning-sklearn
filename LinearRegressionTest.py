# -*- coding: utf-8 -*-
"""
线性回归：LinearRegression，学习得到一个线性模型，尽可能准确预测输出
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, model_selection


"""
加载数据集，测试集大小为原始数据集的0.25倍 ，将数据切分为训练集与测试集
"""


def load_data():
    diabetes = datasets.load_diabetes()
    for i in range(0, 10):
        print(diabetes.target[i])
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                            test_size=0.25, random_state=0)


"""
标准线性回归
def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None）
        参数：
            fit_intercept：布尔值，指定是否计算b
            copy_X：布尔值，是否复制x
            n_jobs：任务并行CPU数目
        属性：
            coef_:Coefficient: 权重向量
            intercept_:intercept： b值
        方法：
            xx.fit()从训练集中学习
            xx.predict() 使用测试集测试
            xx.score()返回预测性能得分
data指定训练样本集、测试样本集、训练样本集对应的标签值、测试样本集对应的标签值
"""


def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                    ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


"""
 岭回归,一种风格正则化方法，在损失函数中加入L2范数惩罚项
   (α||w||)平方的模
   __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
    参数：
        alpha：正则化占比
        fit_intercept：布尔值，指定是否计算b
        copy_X：布尔值，是否复制x
        solver：指定最优化问题的解决算法
            auto：自动
            svd: 使用奇异值分解计算回归系数
        tol:判读迭代是否收敛的阈值
        random_state： 随机生成器
     属性：
        coef_:Coefficient: 权重向量
        intercept_:intercept： b值
     方法：
        xx.fit()从训练集中学习
        xx.predict() 使用测试集测试
        xx.score()返回预测性能得分
    
        
   
"""


def test_Ridge(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                    ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


"""
  检验alpha对预测性能的影响
  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
"""


def test_range_alpha(*args):
    X_train, X_test, y_train, y_test = args
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,
              200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    plt.show()


"""
Lasso,L1正则化 
__init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
    参数：
        max_iter：最大迭代次数
        precompute：布尔值，是否提前计算Game矩阵来加速计算
        warm_start：布尔值，是否使用前一次训练结果继续训练
        selection：
            cyclic：从前向后依次选择权重向量更新
            random：随机选择权重向量更新   
"""


def test_Lasso(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                    ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


def test_Lasso_alpha(*args):
    X_train, X_test, y_train, y_test = args
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,
              200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.show()


"""
  ElaticNet回归是对Lasso回归与岭回归的融合，其惩罚项是L1范数和L2范数的一个权衡
  _init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
  参数：
    l1_ratio ：
    positive： 强制要求权重分量都为正数
  属性：
    n_iter_:实际迭代次数
    
"""


def test_ElaticNet(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print('Coefficients:%s , intercept %.2f' % (regr.coef_, regr.intercept_))
    print("residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test)
                                                ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


def test_ElasticNet_alpha_rho(*args):
    X_train, X_test, y_train, y_test = args
    alphas = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
            regr.fit(X_train, y_train)
            scores.append(regr.score(X_test, y_test))
    # 绘图（曲面图 3D）
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel(r"score")
    ax.set_title("ElasticNet")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train,  y_test = load_data()
    print("普通线性回归")
    test_LinearRegression(X_train, X_test, y_train,  y_test)
    print("岭回归")
    test_Ridge(X_train, X_test, y_train,  y_test)
    # test_range_alpha(X_train, X_test, y_train,  y_test)
    test_Lasso(X_train, X_test, y_train,  y_test)
    # test_Lasso_alpha(X_train, X_test, y_train,  y_test)
    test_ElaticNet(X_train, X_test, y_train,  y_test)
    test_ElasticNet_alpha_rho(X_train, X_test, y_train,  y_test)



