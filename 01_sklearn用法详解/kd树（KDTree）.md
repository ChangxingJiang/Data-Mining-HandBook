# 【sklearn用法详解】kd树(KDTree)

> 包含内容：`sklearn.neighbors.KDTree`
>
> 应用场景：KNN

kd树是一种对k维空间中的实例点进行存储，以便对其进行快速检索的树形数据结构。kd树是二叉树，表示对k维空间的一个划分。构造kd树相当于不断地用垂直于坐标轴的超平面将k维空间切分，构成一系列的k维超矩形区域。kd树的每个结点对应于一个k维超矩形区域。——引自《统计学习方法》

## 常用方法说明

以下参数及返回结果以样本集`[(3, 3), (4, 3), (1, 1), (5, 2)]`为例。

#### KDTree(self, X, leaf_size=40, metric='minkowski', **kwargs)

根据样本点的数据，实例化kd树对象。

* 第1个参数 X：二维数组（`numpy.ndarray`）形式的所有样本点的特征向量。`X[i][j]`表示第i个样本点的第j个特征的值。
* 第2个参数 leaf_size：切换到线性扫描的样本点数量.当样本点少于leaf_size时，将不再使用kd树，而是使用线性扫描。（默认值=40）
* 第3个参数 metric：距离计算方法，详见[【距离计算(DistanceMetric)】](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/01_sklearn%E7%94%A8%E6%B3%95%E8%AF%A6%E8%A7%A3/%E8%B7%9D%E7%A6%BB%E8%AE%A1%E7%AE%97(DistanceMetric).md)。（默认值="minkowski"）

##### 【DEMO】

```python
>>> from sklearn.neighbors import KDTree
>>> KDTree([(3, 3), (4, 3), (1, 1), (5, 2)])
<sklearn.neighbors._kd_tree.KDTree at 0x285e59d0f10>
```

#### get_arrays(self)

返回当前kd树中样本点的基本信息。

* 第1个返回值：二维数组（numpy.ndarray）返回的样本点列表。`data[i][j]`表示第i个样本的第j个特征的值。
* 第2个返回值：一维数组（numpy.ndarray）返回的样本下标。`index[i]`表示第i个样本的下标。
* 第3个返回值：node data（暂不理解）
* 第4个返回值：二维数组（numpy.ndarray）返回的样本点取值范围。`bounds[0][j]`表示第j个特征的最小值，`bounds[1][j]`表示第j个特征的最大值。

##### 【DEMO】

```python
>>> from sklearn.neighbors import KDTree
>>> kdtree = KDTree([(3, 3), (4, 3), (1, 1), (5, 2)])
>>> kdtree.get_arrays()
(array([[3., 3.],
        [4., 3.],
        [1., 1.],
        [5., 2.]]),
 array([0, 1, 2, 3], dtype=int64),
 array([(0, 4, 1, 2.23606798)],
       dtype=[('idx_start', '<i8'), ('idx_end', '<i8'), ('is_leaf', '<i8'), ('radius', '<f8')]),
 array([[[1., 1.]],
        [[5., 3.]]]))
```

#### get_n_calls(self)

返回计算距离的函数被调用的次数。

* 第1个返回值：计算距离的函数被调用的次数

##### 【DEMO】

```python
>>> from sklearn.neighbors import KDTree
>>> kdtree = KDTree([(3, 3), (4, 3), (1, 1), (5, 2)])
>>> kdtree.get_n_calls()
0
>>> kdtree.query([(3, 4)], k=2)
(array([[1.        , 1.41421356]]), array([[0, 1]], dtype=int64))
>>> kdtree.get_n_calls()
4
```

#### reset_n_calls(self)：

将计算距离的函数被调用的次数重置为0。

#### query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False)：

查询距离每一个目标位置最近的k个结点。

* 第1个参数 X：需要查询的结点列表。`X[i]`表示第i个需要查询的目标位置的特征向量。
* 第2个参数 k：需要查询距离每一个目标位置最近结点数量。（默认值=1）
* 第3个参数 return_distance：是否返回结点到目标位置的距离。（默认值=True）
* 第4个参数 dualtree：是否使用广度优先搜索。当dualtree为True时，使用广度优先搜索；否则，使用深度优先搜索。（默认值=False）
* 第5个参数 sort_results：是否依据距离和索引排序结果。（默认值=True）

当return_distance为True时，返回第1个返回值和第2个返回值；否则，只返回第2个返回值。

* 第1个返回值：相邻结点到目标位置的距离。`res1[i][j]`表示第i个需要查询的目标位置到第j个相邻的结点的距离。
* 第2个返回值：相邻结点的索引。`res1[i][j]`表示第i个需要查询的目标位置的第j个相邻的结点的索引。

##### 【DEMO】

```python
>>> from sklearn.neighbors import KDTree
>>> kdtree = KDTree([(3, 3), (4, 3), (1, 1), (5, 2)])
>>> kdtree.query([(3, 4)])
(array([[1.]]), array([[0]], dtype=int64))
>>> kdtree.query([(3, 4)], k=2)  # k>1的情况
(array([[1.        , 1.41421356]]), array([[0, 1]], dtype=int64))
>>> kdtree.query([(3, 4), (5, 6)], k=2)  # X中包含多个位置的情况
(array([[1.        , 1.41421356],
        [3.16227766, 3.60555128]]),
 array([[0, 1],
        [1, 0]], dtype=int64))
>>> kdtree.query([(3, 4)], k=2, return_distance=False)  # return_distance为False的情况
array([[0, 1]], dtype=int64)
```

#### query_radius(self, X, r, return_distance=False, count_only=False, sort_results=False)：

查询每一个目标位置周围半径在r以内的结点。

* 第1个参数 X：同query。
* 第2个参数 r：需要查询的每一个目标位置周围结点的半径。
* 第3个参数 return_distance：同query。（默认值=False）
* 第4个参数 count_only：是否只统计结点数量。当count_only为True时，只返回在半径r以内的结点数量；否则，返回r以内的所有结点。（默认值=False）
* 第5个参数 sort_results：同query。

当count_only为True时，返回第1个返回值；否则，返回结果同query。

* 第1个返回值：每一个目标位置周围半径为r以内的结点数量。`res[i]`表示第i个需要查询的目标位置周围半径r以内的结点数量。

##### 【DEMO】

```python
>>> from sklearn.neighbors import KDTree
>>> kdtree = KDTree([(3, 3), (4, 3), (1, 1), (5, 2)])
>>> kdtree.query_radius([(3, 4)], r=3)
array([array([0, 1, 3], dtype=int64)], dtype=object)
>>> kdtree.query_radius([(3, 4)], r=3, count_only=True)
array([3], dtype=int64)
```

