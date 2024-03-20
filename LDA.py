#!/usr/bin/python
# -*- coding:utf-8 -*-

# 文件名: LDA.py
# 作者: Jamerri
# 创建日期: 2024/3/18
# 版本号: 1.0
# 描述:测试LDA方法

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 训练 LDA 模型
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)

# 获取 LDA 转换后的数据
X_lda = lda.transform(X)

# 绘制 LDA 转换后的数据
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.show()