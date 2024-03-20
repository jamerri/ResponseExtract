#!/usr/bin/python
# -*- coding:utf-8 -*-

# 文件名: PCA.py
# 作者: Jamerri
# 创建日期: 2024/3/18
# 版本号: 1.0
# 描述:测试PCA方法

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
Y = iris.target  # 数据集标签 ['setosa', 'versicolor', 'virginica']，山鸢尾、变色鸢尾、维吉尼亚鸢尾
X = iris.data  # 数据集特征 四维，花瓣的长度、宽度，花萼的长度、宽度

pca = PCA(n_components=2)
pca = pca.fit(X)
X_dr = pca.transform(X)

# 对三种鸢尾花分别绘图
colors = ['red', 'black', 'orange']
# iris.target_names
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[Y == i, 0],
                X_dr[Y == i, 1],
                alpha=1,
                c=colors[i],
                label=iris.target_names[i])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
