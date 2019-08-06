# -*- coding: utf-8 -*-
"""
数据二元化：将数据转换为布尔属性的数值
"""
from sklearn.preprocessing import Binarizer
x = [
    [1, 2, 3, 4],
    [1, 2, 4, 4],
    [2, 3, 2, 3],
    [4, 4, 4, 5]
]
print("before transform:", x)
binaizer = Binarizer(threshold=2.5)
print("after transform:", binaizer.transform(x))

"""
独热码：将非数值属性映射到整数
"""
from sklearn.preprocessing import OneHotEncoder
x = [
    [1, 2, 3, 4],
    [1, 2, 4, 4],
    [2, 3, 2, 3],
    [4, 4, 4, 5]
]
print("before transform:", x)
encoder = OneHotEncoder(sparse=False)
encoder.fit(x)
print("active_features_:", encoder.active_features_)
print("feature_indices_", encoder.feature_indices_)
print("n_values_", encoder.n_values_)
print("", encoder.transform([[1, 2, 3, 4]]))

"""
标准化
MinMaxScaler
"""
from sklearn.preprocessing import MinMaxScaler
x = [
    [1, 2, 3, 4],
    [1, 2, 4, 4],
    [2, 3, 2, 3],
    [4, 4, 4, 5]
]
print("before transform:", x)
scaler = MinMaxScaler(feature_range=(0, 2))
scaler.fit(x)
print("min_is:", scaler.min_)
print("scale_ is", scaler.scale_)
print("data_max is", scaler.data_max_)
print("data_min is", scaler.data_min_)
print("data_range_", scaler.data_range_)
print("after transform:", scaler.transform(x))

"""
正则化:将样本的某个范数缩放到单位1
"""
from sklearn.preprocessing import Normalizer
x = [
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 3, 5, 2, 4],
    [2, 4, 1, 3, 5]
]
print("before transform", x)
normalize = Normalizer(norm="l2")
print("after transform:", normalize.transform(x))
"""
特征选择： 
过滤式选择：先对数据集进行特征选择，然后在训练学习器。特征选择与后续学习器无关
包裹式选择：直接把最终要使用的学习器性能作为特征子集的评价准则。
嵌入式选择：在学习器训练过程中进行特征选择
"""
from sklearn.feature_selection import VarianceThreshold
x = [
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 3, 5, 2, 4],
    [2, 4, 1, 3, 5]
]
selector = VarianceThreshold(1)
selector.fit(x)
print("Variance is %s" %selector.variances_)
print("After transform is %s" %selector.transform(x))
print("The support is %s" %selector.get_support(True))
print("After reverse transform is %s" %
      selector.inverse_transform(selector.transform(x)))