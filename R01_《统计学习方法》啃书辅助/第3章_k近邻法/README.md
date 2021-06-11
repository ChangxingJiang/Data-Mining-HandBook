# 《统计学习方法》啃书辅助：第3章 k近邻法

**k近邻法的学习过程**：没有显示的学习过程。

**k近邻法（用于分类）的预测过程**：在训练数据集中找到与新的输入实例最邻近的k个实例，这k个实例的多数属于某个类，就把该输入实例分为这个类。

**k近邻法的类别划分**：

* 用于解决分类或回归问题的监督学习模型
* 非概率模型：模型取函数形式
* 线性模型：模型函数为线性函数
* 非参数化模型：假设模型参数的维度不固定
* 判别模型：由数据直接学习决策函数$f(X)$

**k近邻法的主要优点**：精度高、对异常值不敏感、无数据输入假定。

**k近邻法的主要缺点**：计算复杂度高、空间复杂度高。

> 【扩展阅读】[sklearn中文文档：1.6 最近邻](https://sklearn.apachecn.org/docs/master/7.html)

-----

#### 用于回归的k近邻法

给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的平均值，就是该输入实例的预测值。

## 3.1： k近邻算法

> 【补充说明】$argmax$函数用于计算因变量取得最大值时对应的自变量的点集。求函数$f(x)$取得最大值时对应的自变量$x$的点集可以写作
>
> $$
> arg \max_{x} f(x)
> $$

## 3.2： k近邻模型

> 【补充说明】**多数表决规则的分类函数**
> $$
> f:R^n \rightarrow \{c_1,c_2,\cdots,c_K\}
> $$
>
> 上式中的“$\rightarrow$”表示映射。分类函数$f$为n维实数向量空间$R^n$到类别集合$\{c_1,c_2,\cdots,c_K\}$的映射。

> **【补充说明】多数表决规则的误分类率**
> $$
> \frac{1}{k} \sum_{x_i \in N_k(x)} I(y_i \ne c_j) = 1 - \frac{1}{k} \sum_{x_i \in N_k(x)} I(y_i=c_j)
> $$
>
> 其中$I$是指示函数，$I(y_i \ne c_j)$当$y_i \ne c_j$时为1，否则为0。

#### $L_p$距离的特征总结（例3-1）

* 当两个向量只有一个维度的值不同时，$L_p$距离的大小与$p$无关；
* 当两个向量有超过一个维度的值不同时，$p$越大，两个向量之间的$L_p$距离越小。

#### 近似误差和估计误差

近似误差：模型估计值与训练数据集的误差，即模型能否准确预测训练数据集。

估计误差：训练数据集与实际数据（测试数据集）的误差，即模型能否准确预测实际数据（测试数据集）。

近似误差减小，模型能够更加地准确预测训练数据集中，但训练数据集中的噪音产生的影响也会增大，估计误差增大，容易过拟合。估计误差减小，模型估计值受更多样本的影响，单个噪声产生的影响也随着缩小，但模型估计值预测训练数据集的准确程度也随之下降，近似误差增大，容易欠拟合。

#### k值的选择

经验规则：k值一般小于训练样本量的平方根

#### Lp距离（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_lp_distance.py)】code.knn.lp_distance

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_lp_distance.py

def lp_distance(p, x1, x2):
    """计算Lp距离

    :param p: [int] 参数p
    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: Lp距离
    """
    n_features = len(x1)
    return pow(sum(pow(abs(x1[i] - x2[i]), p) for i in range(n_features)), 1 / p)
```

#### 欧氏距离（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_eucliean_distance.py)】code.knn.euclidean_distance

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_eucliean_distance.py

def euclidean_distance(x1, x2):
    """计算欧氏距离

    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: 欧氏距离
    """
    n_features = len(x1)
    return pow(sum(pow(x1[i] - x2[i], 2) for i in range(n_features)), 1 / 2)
```

#### 曼哈顿距离（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_manhattan_distance.py)】code.knn.manhattan_distance

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_manhattan_distance.py

def manhattan_distance(x1, x2):
    """计算曼哈顿距离

    :param x1: [tuple/list] 第1个向量
    :param x2: [tuple/list] 第2个向量
    :return: 曼哈顿距离
    """
    n_features = len(x1)
    return sum(abs(x1[i] - x2[i]) for i in range(n_features))
```

## 3.3： k近邻法的实现——kd树

> 【算法3.3 补充说明】在每一次递归中，即使已经将当前结点保存的实例点作为“当前最近点”（3.a），也仍然需要检查另一个子结点对应的区域内是否存在更近点（3.b）。

#### 线性扫描实现的k近邻计算（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_linear_sweep_knn.py)】code.knn.LinearSweepKNN

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_linear_sweep_knn.py

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
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC3%E7%AB%A0_k%E8%BF%91%E9%82%BB%E6%B3%95/%E7%BA%BF%E6%80%A7%E6%89%AB%E6%8F%8F%E5%AE%9E%E7%8E%B0%E7%9A%84k%E8%BF%91%E9%82%BB%E8%AE%A1%E7%AE%97.py)】测试

```python
>>> from code.knn import LinearSweepKNN
>>> from code.knn import euclidean_distance
>>> dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
>>> knn = LinearSweepKNN(dataset[0], dataset[1], k=2, distance_func=euclidean_distance)
>>> knn.count((3, 4))
1
```

#### 搜索kd树的停止条件

如果父结点的另一个子结点的超矩形区域与超球体不相交，只能说明是另一个子结点在被用于切分的维度上，在当前方向上，不再会有必当前最近点更近的点，但并不能说明在当前维度的另一个方向上，以及在其他方向上并不存在比当前最近点更近的点。例如例3.3中，F的区域的超矩形与超球体已不想交，但之后的搜索中还能找到更近的最近点E。

这个剪枝条件更应该被描述为：若当前结点距离目标位置的距离，在当前方向的分量上都已经超过超球体的半径，则不用考虑另一个子结点的情况。

#### kd树的抽象数据类型

kd树是存储k维空间数据的树形数据结构，并支持快速地近邻搜索。从形式上来说，kd树是将每一个元素存在一个结点上，kd树的抽象数据类型（ADT）支持以下访问方法，用T表示这一ADT实例：

* T(data)：构造kd树实例。在Python中，我们用`__init__`这个特殊方法来实现它。
* T._build_kd_tree(data)：根据k维空间数据data构造kd树，并返回构造的kd树的根结点。
* T.search_nearest(x)：返回x的最近邻点。如果kd树p中没有元素，这个操作将出错。
* T.search_knn(x,k)：返回距离x最近的k个点。如果kd树P中元素的数量不足k，这个操作这返回所有的点。
* len(P)：返回kd树P中元素的数量。在Python中，我们用`__len__`这个特殊方法来实现它。

#### KD树（原生Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_kd_tree.py)】code.knn.KDTree

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_kd_tree.py

import heapq

class KDTree:
    class _Node:
        """kd树的轻量级结点"""
        __slots__ = "element", "axis", "left", "right"

        def __init__(self, element, axis=0, left=None, right=None):
            self.element = element  # 当前结点的值
            self.axis = axis  # 当前结点用于切分的轴
            self.left = left  # 当前结点的左子结点
            self.right = right  # 当前结点的右子结点

        def __lt__(self, other):
            """定义_Node之间的小于关系（避免heapq比较大小报错）"""
            return self.element < other.element

    def __init__(self, data, distance_func):
        """构造平衡kd树实例"""
        self._size = len(data)  # 元素总数
        self._distance_func = distance_func  # 用于计算距离的函数
        if self._size > 0:
            self._dimension = len(data[0])  # 计算输入数据的空间维度数
            self._root = self._build_kd_tree(data, depth=0)  # kd树的根结点
        else:
            self._dimension = 0
            self._root = None

    def _build_kd_tree(self, data, depth):
        """根据输入数据集data和当前深度depth，构造是平衡kd树"""
        if not data:
            return None

        # 处理当前结点数据
        select_axis = depth % self._dimension  # 计算当前用作切分的坐标轴
        median_index = len(data) // 2  # 计算中位数所在坐标
        data.sort(key=lambda x: x[select_axis])  # 依据需要用作切分的坐标轴排序输入的数据集

        # 构造当前结点
        node = self._Node(data[median_index], axis=select_axis)
        node.left = self._build_kd_tree(data[:median_index], depth + 1)  # 递归构造当前结点的左子结点
        node.right = self._build_kd_tree(data[median_index + 1:], depth + 1)  # 递归构造当前结点的右子结点
        return node

    def search_nn(self, x):
        """返回x的最近邻点"""
        return self.search_knn(x, 1)

    def search_knn(self, x, k):
        """返回距离x最近的k个点"""
        res = []
        self._search_knn(res, self._root, x, k)
        return [(node.element, -distance) for distance, node in sorted(res, key=lambda xx: -xx[0])]

    def _search_knn(self, res, node, x, k):
        if node is None:
            return

        # 计算当前结点到目标点的距离
        node_distance = self._distance_func(node.element, x)

        # 计算当前结点到目标点的距离（在当前用于划分的维度上）
        node_distance_axis = self._distance_func([node.element[node.axis]], [x[node.axis]])

        # [第1步]处理当前结点
        if len(res) < k:
            heapq.heappush(res, (-node_distance, node))
        elif node_distance_axis < (-res[0][0]):
            heapq.heappushpop(res, (-node_distance, node))

        # [第2步]处理目标点所在的子结点
        if x[node.axis] <= node.element[node.axis]:
            self._search_knn(res, node.left, x, k)
        else:
            self._search_knn(res, node.right, x, k)

        # [第3步]处理目标点不在的子结点
        if len(res) < k or node_distance_axis < (-res[0][0]):
            if x[node.axis] <= node.element[node.axis]:
                self._search_knn(res, node.right, x, k)
            else:
                self._search_knn(res, node.left, x, k)

    def __len__(self):
        """返回kd树P中元素的数量"""
        return self._size
```

#### 基于kd树实现的k近邻计算（sklearn.neighbors.KDTree实现）

【延伸知识】[sklearn.neighbors.KDTree用法详解](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/01_sklearn%E7%94%A8%E6%B3%95%E8%AF%A6%E8%A7%A3/kd%E6%A0%91%EF%BC%88KDTree%EF%BC%89.md)

【延伸知识】[sklearn.neighbors.DistanceMetric用法详解](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/01_sklearn%E7%94%A8%E6%B3%95%E8%AF%A6%E8%A7%A3/%E8%B7%9D%E7%A6%BB%E8%AE%A1%E7%AE%97(DistanceMetric).md)

【官方API文档】[sklearn.neighbors.KDTree官方API文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree)

【官方API文档】[sklearn.neighbors.DistanceMetric官方API文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric)

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_kd_tree_knn.py)】code.knn.KDTreeKNN

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_kd_tree_knn.py

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
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC3%E7%AB%A0_k%E8%BF%91%E9%82%BB%E6%B3%95/%E5%9F%BA%E4%BA%8Ekd%E6%A0%91%E5%AE%9E%E7%8E%B0%E7%9A%84k%E8%BF%91%E9%82%BB%E8%AE%A1%E7%AE%97.py)】测试

```python
>>> from code.knn import KDTreeKNN
>>> dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
>>> knn = KDTreeKNN(dataset[0], dataset[1], k=2)
>>> knn.count((3, 4))
1
```

#### 简单交叉验证计算k最优的KNN分类器（sklearn.neighbors.KNeighborsClassifier实现）

【官方API文档】[sklearn.neighbors.KNeighborsClassifier官方API文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_build_best_knn_simple_cross_validation.py)】code.knn.build_best_knn_simple_cross_validation

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_build_best_knn_simple_cross_validation.py

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def build_best_knn_simple_cross_validation(x, y):
    """简单交叉验证计算k最优的KNN分类器"""
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)  # 拆分训练集&验证集(80%)和测试集(20%)
    x11, x12, y11, y12 = train_test_split(x1, y1, test_size=0.25, random_state=0)  # 拆分训练集(60%)和验证集(20%)
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x11, y11)
        score = knn.score(x12, y12)
        if score > best_score:
            best_k, best_score = k, score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x1, y1)
    return best_k, best_knn.score(x2, y2)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC3%E7%AB%A0_k%E8%BF%91%E9%82%BB%E6%B3%95/%E7%AE%80%E5%8D%95%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E8%AE%A1%E7%AE%97k%E6%9C%80%E4%BC%98%E7%9A%84KNN%E5%88%86%E7%B1%BB%E5%99%A8.py)】测试

```python
>>> from sklearn.datasets import make_blobs
>>> from code.knn import build_best_knn_simple_cross_validation
>>> # 生成随机样本数据
    X, Y = make_blobs(n_samples=1000, n_features=10, centers=5,
                      cluster_std=5000, center_box=(-10000, 10000), random_state=0)
>>> # 计算k最优的KNN分类器
    final_k, final_score = build_best_knn_simple_cross_validation(X, Y)
>>> final_k
75
>>> final_score
0.900
```

#### S折交叉验证计算k最优的KNN分类器（sklearn.model_selection.cross_val_score实现）

【官方API文档】[sklearn.model_selection.cross_val_score官方API文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross_val_score#sklearn.model_selection.cross_val_score)

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_build_best_knn_s_fold_cross_validation.py)】code.knn.build_best_knn_s_fold_cross_validation

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/knn/_build_best_knn_s_fold_cross_validation.py

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def build_best_knn_s_fold_cross_validation(x, y):
    """S折交叉验证计算k最优的KNN分类器"""
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)  # 拆分训练集(80%)和测试集(20%)
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x1, y1, cv=10, scoring="accuracy")
        score = scores.mean()
        if score > best_score:
            best_k, best_score = k, score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x1, y1)
    return best_k, best_knn.score(x2, y2)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC3%E7%AB%A0_k%E8%BF%91%E9%82%BB%E6%B3%95/S%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E8%AE%A1%E7%AE%97k%E6%9C%80%E4%BC%98%E7%9A%84KNN%E5%88%86%E7%B1%BB%E5%99%A8.py)】测试

```python
>>> from sklearn.datasets import make_blobs
>>> from code.knn import build_best_knn_s_fold_cross_validation
>>> # 生成随机样本数据
    X, Y = make_blobs(n_samples=1000, n_features=10, centers=5,
                      cluster_std=5000, center_box=(-10000, 10000), random_state=0)
>>> # 计算k最优的KNN分类器
    final_k, final_score = build_best_knn_s_fold_cross_validation(X, Y)
>>> final_k
74
>>> final_score
0.905
```

## 延伸阅读

【官方API文档】[随机数据集(datasets.make...)的官方API文档](https://scikit-learn.org/stable/modules/classes.html#samples-generator)

【官方API文档】[经典数据集(datasets.load...)的官方API文档](https://scikit-learn.org/stable/modules/classes.html#loaders)

首先，讨论k近邻法对随机数据集的准确率。随机数据集共包含10000个样本，所有随机使用随机种子为0。随机地将数据集切分为两部分，分别为训练集（8000个样本）和测试集（2000个样本）。采用S折交叉验证的方法选择最优k值。

| 随机数据集形态 (省略随机种子)                                | 最优k | 最优k的准确率 |
| ------------------------------------------------------------ | ----- | ------------- |
| 各向同性的斑点状数据 : `make_blobs(n_samples=1000, n_features=10, centers=2, cluster_std=5000, center_box=(-10000, 10000), random_state=0)` | 58    | 0.980         |
| 各向同性的斑点状数据 : `make_blobs(n_samples=1000, n_features=10, centers=5, cluster_std=5000, center_box=(-10000, 10000), random_state=0)` | 74    | 0.905         |
| 各向同性的斑点状数据 : `make_blobs(n_samples=1000, n_features=10, centers=10, cluster_std=5000, center_box=(-10000, 10000), random_state=0)` | 32    | 0.900         |
| 有噪声的同心环 : `make_circles(n_samples=1000, noise=0.1, random_state=0)` | 18    | 0.825         |
| 有噪声的同心环 : `make_circles(n_samples=1000, noise=0.2, random_state=0)` | 62    | 0.670         |
| 有噪声的两个月形半圆 : `make_moons(n_samples=1000, noise=0.2, random_state=0)` | 26    | 0.970         |
| 有噪声的两个月形半圆 : `make_moons(n_samples=1000, noise=0.4, random_state=0)` | 9     | 0.865         |

接着，讨论k近邻法对经典数据集的准确率。所有随机使用随机种子为0。随机地将数据集切分为两部分，分别为训练集（2/3的样本）和测试集（1/3的样本）。采用S折交叉验证的方法选择最优k值。

| 经典数据集                                      | 最优k | 最优k的准确率 |
| ----------------------------------------------- | ----- | ------------- |
| 鸢尾花数据集 : `load_iris()`                    | 5     | 0.98          |
| 8×8数字数据集 : `load_digits()`                 | 1     | 0.9850        |
| 威斯康星州乳腺癌数据集 : `load_breast_cancer()` | 8     | 0.9579        |
| 葡萄酒数据集 : `load_wine()`                    | 1     | 0.7667        |

