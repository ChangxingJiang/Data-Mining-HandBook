import collections

from sklearn.neighbors import KDTree


class KDTreeKNN:
    """kd实现的k近邻计算"""

    def __init__(self, x, y, k, metric="euclidean"):
        self.x, self.y, self.k = x, y, k
        self.kdtree = KDTree(self.x, metric=metric)  # 构造KD树

    def count(self, x):
        """计算实例x所属的类y"""
        index = self.kdtree.query([x], self.k, return_distance=False)
        count = collections.Counter()
        for i in index[0]:
            count[self.y[i]] += 1
        return count.most_common(1)[0][0]
