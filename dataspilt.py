# # -*- coding: utf-8 -*-
# """
# 数据集切分
# """
# from sklearn.model_selection import train_test_split
# x = [
#     [1, 2, 3, 4],
#     [11, 12, 13, 14],
#     [21, 22, 23, 24],
#     [31, 32, 33, 34],
#     [41, 42, 43, 44],
#     [51, 52, 53, 54],
#     [61, 62, 63, 64],
#     [71, 72, 73, 74]
# ]
# y = [1, 1, 0, 0, 1, 1, 0, 0]
# # 非分层采样
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
# print("x_train=", x_train)
# print("y_train=", y_train)
# print("x_test=", x_test)
# print("y_test=", y_test)
# # 分层采样 保证训练集和测试集中各类样本的比例与原始数据一致
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)
# print("x_train=", x_train)
# print("y_train=", y_train)
# print("x_test=", x_test)
# print("y_test=", y_test)
"""
k折交叉切分, k-1折作为测试集，shuffle切分前进行数据混洗
"""
# from sklearn.model_selection import KFold
# import numpy as np
# x = np.array([
#     [1, 2, 3, 4],
#     [11, 12, 13, 14],
#     [21, 22, 23, 24],
#     [31, 32, 33, 34],
#     [41, 42, 43, 44],
#     [51, 52, 53, 54],
#     [61, 62, 63, 64],
#     [71, 72, 73, 74],
#     [81, 82, 83, 84]
# ])
# y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1])
# folder = KFold(n_splits=3, random_state=0, shuffle=False)
# for train_index, test_index in folder.split(x, y):
#     print("Train_index=", train_index)
#     print("Test-=", test_index)
#     print("x_train=", x[train_index])
#     print("x_test=", x[test_index])
#     print(" ")
# # shuffle: 混洗数据
# shuffle_folder = KFold(n_splits=3, random_state=0, shuffle=True)
# for train_index, test_index in shuffle_folder.split(x, y):
#     print(" shuffle Train_index=", train_index)
#     print("shuffle Test-=", test_index)
#     print("shuffle x_train=", x[train_index])
#     print("shuffle x_test=", x[test_index])
#     print(" ")
"""
分层k折交叉切分
"""
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
x = np.array([
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34],
    [41, 42, 43, 44],
    [51, 52, 53, 54],
    [61, 62, 63, 64],
    [71, 72, 73, 74]
])
y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
folder = KFold(n_splits=4, random_state=0, shuffle=False)
startified_folder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)
for train_index, test_index in folder.split(x, y):
    print("Train_index=", train_index)
    print("Test-=", test_index)
    print("y_train=", y[train_index])
    print("y_test=", y[test_index])
    print(" ")
for train_index, test_index in startified_folder.split(x, y):
    print("startified Train_index=", train_index)
    print("startified Test-=", test_index)
    print("startified y_train=", y[train_index])
    print("startified y_test=", y[test_index])
    print(" ")

