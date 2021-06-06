# 【sklearn用法详解】距离计算(DistanceMetric)

> 包含内容：`sklearn.neighbors.DistanceMetric`及其子类
>
> 应用场景：kd树、聚类等用到距离的方法的距离计算

每一种不同的距离计算方法，都有唯一的距离名称（string identifier），例如`euclidean`、`hamming`等；以及对应的距离计算类，例如`EuclideanDistance`、`HammingDistance`等。这些距离计算类都是`DistanceMetric`的子类。

这些距离计算类可以通过父类`DistanceMetric`的类方法`get_metric`，使用距离名称实例化：

```python
>>> from sklearn.neighbors import DistanceMetric
>>> DistanceMetric.get_metric("euclidean")
<sklearn.neighbors._dist_metrics.EuclideanDistance at 0x228046ab740>
```

**不建议**直接导入非公开的子类：

```python
>>> from sklearn.neighbors._dist_metrics import EuclideanDistance
>>> EuclideanDistance()
<sklearn.neighbors._dist_metrics.EuclideanDistance at 0x228046ab890>
```

有参数的距离计算方法，可以通过关键字参数的方式实例化：

```python
>>> from sklearn.neighbors import DistanceMetric
>>> dist1 = DistanceMetric.get_metric("euclidean")
>>> dist1.pairwise([[0, 1, 2], [3, 4, 5]])
array([[0.        , 5.19615242],
       [5.19615242, 0.        ]])
>>> dist2 = DistanceMetric.get_metric("minkowski", p=2)  # Euclidean distance 即当p=2时的 Minkowski distance
>>> dist2.pairwise([[0, 1, 2], [3, 4, 5]])
array([[0.        , 5.19615242],
       [5.19615242, 0.        ]])
```

在sklearn中，在绝大部分需要用到距离的场景，既可以直接将距离计算类实例化的对象作为参数；也可以将距离名称作为参数，由sklearn自动实例化对应的距离计算类。

## sklearn的距离计算方法

在元素值为实数的情况下，计算向量之间的距离的方法。

| 距离名称    | 常用中文名                     | 距离计算类          | 参数                  | 距离计算公式                           |
| ----------- | ------------------------------ | ------------------- | --------------------- | -------------------------------------- |
| euclidean   | 欧氏距离                       | EuclideanDistance   | -                     | `sqrt(sum((x - y)^2))`                 |
| manhattan   | 曼哈顿距离                     | ManhattanDistance   | -                     | `sum(|x - y|)`                         |
| chebyshev   | 切比雪夫距离                   | ChebyshevDistance   | -                     | `max(|x - y|)`                         |
| minkowski   | 闵式距离/闵可夫斯基距离/Lp距离 | MinkowskiDistance   | p                     | `sum(|x - y|^p)^(1/p)`                 |
| wminkowski  | 加权闵可夫斯基距离             | WMinkowskiDistance  | p,w                   | `sum(|w * (x - y)|^p)^(1/p)`           |
| seuclidean  | 标准化欧氏距离                 | SEuclideanDistance  | V                     | `sqrt(sum((x - y)^2 / V))`             |
| mahalanobis | 马氏距离                       | MahalanobisDistance | Σ(协方差矩阵),μ(均值) | `sqrt((x - μ)' Σ^{-1} (x - μ))`        |
| hamming     | 汉明距离*                      | HammingDistance     | -                     | `x和y不同的维度数/总维度数`            |
| canberra    |                                | CanberraDistance    | -                     | `sum(|x - y|/ (|x|+ |y|))`             |
| braycurtis  |                                | BrayCurtisDistance  | -                     | `sum(|x - y|) / (sum(|x|) + sum(|y|))` |

> 注：这里的汉明距离是除以了总维度数的，与我们通常理解的不除以总维度数的汉明距离不同。

适用于计算通过纬度和经度确定的位置之间的距离的方法。输入和输出均以弧度为单位。

| 距离名称  | 常用中文名 | 距离计算类        | 距离计算公式                                                 |
| --------- | ---------- | ----------------- | ------------------------------------------------------------ |
| haversine |            | HaversineDistance | `2*arcsin(sqrt(sin^2(|x2-x1|/2) + cos(x1)cos(x2)sin^2(|y2-y1|/2)))​` |

在元素值只有布尔值的情况下，计算向量之间的距离的方法。先定义以下变量：

* N：向量维度数
* NTT：两个向量均为True的维度数
* NTF：第1个向量为True，第2个向量为False的维度数
* NFT：第1个向量为False，第2个向量为True的维度数
* NFF：两个向量均为False的维度数
* NNEQ：两个向量不同的维度数，NNEQ = NTF + NFT
* NNZ：有一个向量不为0的维度数，NNZ = NTF + NFT + NTT

| 距离名称       | 常用中文名 | 距离计算类             | 距离计算公式                    |
| -------------- | ---------- | ---------------------- | ------------------------------- |
| jaccard        | 杰卡德距离 | JaccardDistance        | `NNEQ / NNZ`                    |
| matching       | 匹配距离   | MatchingDistance       | `NNEQ / N`                      |
| dice           |            | DiceDistance           | `NNEQ / (NTT + NNZ)`            |
| kulsinski      |            | KulsinskiDistance      | `(NNEQ + N - NTT) / (NNEQ + N)` |
| rogerstanimoto |            | RogersTanimotoDistance | `2 * NNEQ / (N + NNEQ)`         |
| russellrao     |            | RussellRaoDistance     | `(N - NTT) / N`                 |
| sokalmichener  |            | SokalMichenerDistance  | `2 * NNEQ / (N + NNEQ)`         |
| sokalsneath    |            | SokalSneathDistance    | `NNEQ / (NNEQ + 0.5 * NTT)`     |

## 常用方法说明

#### get_metric(cls, *args, **kwargs)

根据字符串格式的距离名称实例化对应的距离计算类。如果距离计算中需要用到其他参数，则使用关键字参数的形式传递这些参数。

当参数特殊，使当前距离计算方法可以转换为其他更简单的距离计算方法时，则自动实例化更简单的距离计算类。例如p=2时的闵式距离将自动转换为欧氏距离类。

* 第1个参数（metric）：距离名称
* 关键字参数：距离计算中需要的参数

##### 【DEMO】

```python
>>> from sklearn.neighbors import DistanceMetric
>>> DistanceMetric.get_metric("euclidean")
<sklearn.neighbors._dist_metrics.EuclideanDistance at 0x228046ab740>
>>> DistanceMetric.get_metric("minkowski", p=2)  # Euclidean distance 即当p=2时的 Minkowski distance
<sklearn.neighbors._dist_metrics.EuclideanDistance at 0x228047233c0>
```

#### pairwise(self, *args, **kwargs)

输入二维列表，其中每一行为一个向量，计算所有向量之间的距离；返回二维数组（`numpy.ndarray`），其中`res[i][j]`表示第i个向量与第j个向量的距离。

* 第1个参数（X）：二维列表

##### 【DEMO】

```python
>>> from sklearn.neighbors import DistanceMetric
>>> dist = DistanceMetric.get_metric("euclidean")
>>> dist.pairwise([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
array([[ 0.        ,  5.19615242, 10.39230485],
       [ 5.19615242,  0.        ,  5.19615242],
       [10.39230485,  5.19615242,  0.        ]])
```

