# -*- coding: utf-8 -*-
from sklearn.metrics import zero_one_loss, log_loss

y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
# 0-1损失函数 normalize=True 返回分类错误的比率  返回分类错误的数目
print("zero_one_loss<fraction>:", zero_one_loss(y_true, y_pred, normalize=True))
print("zero_one_loss<num>:", zero_one_loss(y_true, y_pred, normalize=False))

x_true = [1, 1, 1, 0, 0, 0]
x_predict = [
    [0.1, 0.9],
    [0.2, 0.8],
    [0.3, 0.7],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.9, 0.1]
]
# 对数损失函数 True：返回所有样本对数损失的均值 False： 返回样本对数损失的总和
print("log_loss<average>:", log_loss(y_true, y_pred, normalize=True))
print("log_loss<toatl>:", log_loss(y_true, y_pred, normalize=False))
