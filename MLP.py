# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

"""
生成线性可分数据集的生成算法:
    在z轴坐标为20的上方生成n个随机点作为正类，在z轴坐标为10的下方生成n个随机数作为负类。
在平面z = 10, z=20作为隔离带。然后45旋转x轴。
"""


def creat_data(n):
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1))
    x_13 = 20 + np.random.randint(0, 10, (n, 1,))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 10 - np.random.randint(0, 10, (n, 1,))

    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_13 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2
    new_x_23 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2

    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    minus_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])
    samples = np.vstack([plus_samples, minus_samples])
    np.random.shuffle(samples)  # 混洗数据
    return samples


"""
绘制数据集的函数
"""


def plot_samples(ax, samples):
    Y = samples[:, -1]
    Y = samples[:, -1]
    position_p = y == 1  # 正类位置
    position_m = y == -1  # 父类位置
    ax.scatter(samples[position_p, 0], samples[position_p, 1],
               samples[position_p, 2], marker='+', label='+', color='b')
    ax.scatter(samples[position_m, 0], samples[position_m, 1],
               samples[position_m, 2], marker='^', label='-', color='y')


"""
感知机学习算法的原始形式算法函数
"""


def perception(train_data, eta, w_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    w = w_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while i < length:
            step_num += 1
            x_i = x[i].reshape((x.shape[1], 1))
            y_i = y[i]
            if y_i * (np.dot(np.transpose(w), x_i) + b) <= 0:
                w = w + eta * y_i * x_i
                b = b + eta * y_i
                break
            else:
                i = i + 1
        if i == length:
            break
    return w, b, step_num


def creat_hyperplane(x, y, w, b):
    return (-w[0][0] * x - w[1][0] * y - b)/[2][0]

data = creat_data(100)
eta, w_0, b_0 = 0.1, np.ones((3, 1), dtype=float), 1
w, b, num = perception(data, eta, w_0, b_0)

fig = plt.figure()
plt.subtitle("perception")
ax = Axes3D(fig)
plot_samples(ax, data)

x = np.linspace(-30, 100, 100)
y = np.linspace(-30, 100, 100)
x, y = np.meshgrid(x, y)
z = creat_hyperplane(x, y, w, b)
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='g', alpha=0.2)
ax.legend(loc="best")
plt.show()