# -*- coding: utf-8 -*-
"""
性能度量
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
accuracy_score()
# 计算分类结果准确率  简单比较y_pred 与 y_true 的不同样本个数
precision_score()
# 计算分类结果查准率  precision_score， 预测为正类的样本中真正正类的个数
recall_score()
# 计算分类结果查全率，真实的正类中，有多少比例被预测为正类
f1_score()
# 计算分类结果的F1值 ，查准率与查全率的调和均值
fbeta_score()
# 计算分类结果的beta值

