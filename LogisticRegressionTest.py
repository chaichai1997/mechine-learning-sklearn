# -*- coding: utf-8 -*-
"""
逻辑回归：LogisticRegression
    属于分类模型，不仅预测类别，还可得到近似概率预测
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, model_selection


"""
    数据获取
    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'],
                 filename=iris_csv_filename)
    data: 属性值
    terget：标签[0/1/2]
    target_names:标签对应种类名称
    feature_names：属性名称
    filename：文件名称
"""


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    for i in range(0, 10):
        print(iris.data[i], iris.target[i])
    return model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                            random_state=0, stratify=y_train)


"""
    Logistic Regression
    __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None):
        参数：
            penalty：字符串，指定正则化策略
            dual：布尔值，False求解原始形式，True求解对偶形式
            tol：收敛阈值
            C：惩罚系数的导数，值越小，正则化项越大
            fit_intercept：布尔值，是否需要计算b
            class_weight：一个字典或“balanced”
                字典给出每个分类的权重
                未指定则每个分类的权重都为1
                balanced 权重与样本出现频率成反比
            random_state：随机生成器
            solver:字符串，指定最优化问题的求解算法
            multi_class：用于多分类问题的策略
        属性：
        coef_:Coefficient: 权重向量
        intercept_:intercept： b值
        n_iter_:实际迭代次数
        方法：
            xx.fit()从训练集中学习
            xx.predict() 使用测试集测试
            xx.predict_log_proba(x):返回一个数组，返回x预测为各个类别的概率的对数值
            xx.predict_proba(x):返回一个数组，返回x预测为各个类别的概率
            xx.score()返回预测性能得分         
"""


def test_LogistcRegression(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print("Coefficient: %s,  intercept %s" % (regr.coef_, regr.intercept_))
    print("score: %.2f" % regr.score(X_test, y_test))


"""
    测试mult_class对分类性能的影响（多类分类）
    只有solver为牛顿法或拟牛顿才可配合multinomial
"""


def test_LogisticRegression_multinomial(*args):
    X_train, X_test, y_train, y_test = args
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, y_train)
    print("Coefficient: %s,  intercept %s" % (regr.coef_, regr.intercept_))
    print("score: %.2f" % regr.score(X_test, y_test))


""""
   测试正则化系数对分类的影响
   np.logspace:返回在对数刻度上均匀间隔的数字。
"""


def test_LogisticRegression_C(*args):
    X_train, X_test, y_train, y_test = args
    Cs = np.logspace(-2, 4, num=100)
    scores = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Logistic Regression")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_LogistcRegression(X_train, X_test, y_train, y_test)
    # test_LogisticRegression_multinomial(X_train, X_test, y_train, y_test)
    test_LogisticRegression_C(X_train, X_test, y_train, y_test)
