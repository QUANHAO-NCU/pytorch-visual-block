"""
参考自
https://zhuanlan.zhihu.com/p/336501183
"""

import numpy as np
import math
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.cluster import dbscan

X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
xx = [i[0] for i in X]
yy = [i[1] for i in X]
plt.scatter(xx, yy, s=100, alpha=0.6)
plt.show()
# eps为邻域半径，min_samples为最少点数目
core_samples, cluster_ids = dbscan(X, eps=0.2, min_samples=10)

"""
eps: 距离阈值，当两个点的距离小于这个值时，认为其中一个点再另一个点的邻域内
min_samples: 一个点的邻域内至少得有这么多个点，才能把这个点视为核心点
Points:要进行DBSCAN分析的点集
算法步骤：
1、遍历所有点，当点满足成为核心点的步骤后，将该点加入核心点，记为类别k，并且将这个核心点邻域eps内的所有点标记为类别k
2、遍历所有点，若一个点既属于类别k1，又属于类别k2，又属于类别k3...则将类别k1,k2,k3合并为一个类别。
3、完成聚类，返回聚类结果。
"""


def Euclidean_distance(coordinate1, coordinate2):
    return math.sqrt((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2)


def DBSCAN(eps, min_samples, Points: list, distFunc):
    # 每个点自动编号，编号为其下标
    # 对Points排序，便于遍历操作，降低复杂度 按照横坐标x排序
    Points.sort(key=lambda x: x[0])
    # 核心点 数据类型->id:[point1,point2]
    core_points = {}
    # 标记每个点属于哪个类别 数据类型->item:[k1,k2,k3]
    points_cluster = [[] for i in Points]
    for pointId, point in enumerate(Points):
        core_points[pointId] = []
        # 自成一类
        points_cluster[pointId].append(pointId)
        count = 0
        for otherPointId, otherPoint in enumerate(Points):
            # 实际中度量点之间的距离不一定是欧氏距离
            dist = distFunc(point, otherPoint)
            if dist < eps:
                core_points[pointId].append(otherPointId)
                count += 1
        if count < min_samples:
            # 邻域内的点数不够
            del core_points[pointId]
        else:
            # 把点标记为类别k
            for otherPointId in core_points[pointId]:
                points_cluster[otherPointId].append(pointId)

    for pointId, point in enumerate(Points):
        mergeCluster = points_cluster[pointId]

        pass
    pass


if __name__ == '__main__':
    X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
    points = [[i[0], i[1]] for i in X]

    DBSCAN(0.2, 5, points, Euclidean_distance)
