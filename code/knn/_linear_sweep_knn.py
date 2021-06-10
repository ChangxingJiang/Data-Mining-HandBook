import collections
import heapq


class LinearSweepKNN:
    """线性扫描实现的k近邻计算"""

    def __init__(self, x, y, k, distance_func):
        self.x, self.y, self.k, self.distance_func = x, y, k, distance_func

    def count(self, x):
        """计算实例x所属的类y
        时间复杂度：O(N+KlogN) 线性扫描O(N)；自底向上构建堆O(N)；每次取出堆顶元素O(logN)，取出k个共计O(KlogN)
        """
        n_samples = len(self.x)
        distances = [(self.distance_func(x, self.x[i]), self.y[i]) for i in range(n_samples)]
        heapq.heapify(distances)
        count = collections.Counter()
        for _ in range(self.k):
            count[heapq.heappop(distances)[1]] += 1
        return count.most_common(1)[0][0]
