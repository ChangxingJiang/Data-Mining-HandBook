# 《统计学习方法》啃书辅助：第5章 决策树

**决策树对数据的要求**：

**决策树的学习过程**：利用训练数据，根据损失函数最小化的原则建立决策树模型。学习通常包括特征选择、决策树的生成、决策树的修剪三个步骤。

**决策树的预测过程**：对新的数据，利用决策树模型进行分类。

**决策树的类别划分**：

* 用于解决分类和回归问题的监督学习模型
* 概率模型：模型取条件概率分布形式$P(y|x)$
* 非参数化模型：假设模型参数的维度不固定
* 判别模型：由数据直接学习决策函数$f(X)$

**决策树的主要优点**：模型具有可读性，分类速度快。

**决策树的主要缺点**：

> 【扩展阅读】[sklearn中文文档：1.10 决策树](https://sklearn.apachecn.org/docs/master/11.html)

## 5.1 决策树模型与学习

> **【名词解释】划分 （以下定义来自浙江大学《概率论与数理统计》第四版 P. 17）**
>
> 设S为试验E的样本空间，$B_1,B_2,\cdots,B_n$为E的一组事件。若
>
> 1. $B_i B_j = \varnothing$，$i \ne j$，$i,j=1,2,\cdots,n$；
> 2. $B_1 \cup B_2 \cup \cdots \cup B_n = S$，
>
> 则称$B_1,B_2,\cdots,B_n$为样本空间S的一个划分。

> 【补充说明】图5.2 (b) 中左下角区域的条件概率分布似乎应为0。

#### NP完全问题

【扩展阅读】[什么是P问题、NP问题和NPC问题 - Matrix67](http://www.matrix67.com/blog/archives/105)

> **【什么是P问题、NP问题和NPC问题 - Matrix67】摘要**
>
> **多项式时间** 我们可以将时间复杂度分为两类；第一类我们称为多项式级的复杂度，其规模n出现在底数的位置，例如$O(1)$、$O(logN)$、$O(N^a)$等；第二类我们称为非多项式级的复杂度，例如$O(a^n)$、$O(N!)$等。我们将第一类时间复杂度称为多项式时间。
>
> **P问题** 如果一个问题可以找到一个能在多项式的时间里解决它的算法，那么这个问题就属于P问题。
>
> **NP问题** NP问题有两种定义。第一种定义：如果一个问题可以在多项式的时间内验证一个解的问题，那么这个问题就属于NP问题。第二种定义：如果一个问题可以在多项式的时间内踩出一个解，那么这个问题就属于NP问题。
>
> **约化** 一个问题A可以约化为问题B的含义是，可以用问题B的解法解决问题A。一般来说，B的时间复杂度高于或等于A的时间复杂度。另外，约化具有传递性，如果问题A可约化为问题B，问题B可约化为问题C，则问题A一定可约化为问题C。
>
> **NPC问题** NPC问题定义为同时满足如下两个条件的问题。首先，它必须是一个NP问题；然后，所有的NP问题都可以约化到它。在现阶段，我们可以直观地理解，NPC问题目前没有多项式的有效算法，只能用指数级甚至阶乘级复杂度的搜索。
>
> **NP-Hard问题** NP-Hard问题不一定是一个NP问题，但是所有的NP问题都可以约化到它。

## 5.2 特征选择

> 【补充说明】训练数据集D关于特征A的值的熵$H_A(D)$即特征A的熵。

#### 例5.1数据集

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/example/_li.py)】code.example.load_li_5_1

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/example/_li.py

import numpy as np

def load_li_5_1():
    """《统计学习方法》李航 例5.1 P.71"""
    return [np.array([["青年", "否", "否", "一般"],
                      ["青年", "否", "否", "好"],
                      ["青年", "是", "否", "好"],
                      ["青年", "是", "是", "一般"],
                      ["青年", "否", "否", "一般"],
                      ["中年", "否", "否", "一般"],
                      ["中年", "否", "否", "好"],
                      ["中年", "是", "是", "好"],
                      ["中年", "否", "是", "非常好"],
                      ["中年", "否", "是", "非常好"],
                      ["老年", "否", "是", "非常好"],
                      ["老年", "否", "是", "好"],
                      ["老年", "是", "否", "好"],
                      ["老年", "是", "否", "非常好"],
                      ["老年", "否", "否", "一般", "否"]]),
            np.array(["否", "否", "是", "是", "否",
                      "否", "否", "是", "是", "是",
                      "是", "是", "是", "是", "否"])]
```

#### 熵（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_entropy.py)】code.dicision_tree.entropy

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_entropy.py

import collections
from math import log

def entropy(y, base=2):
    """计算随机变量Y的熵"""
    count = collections.Counter(y)
    ans = 0
    for freq in count.values():
        prob = freq / len(y)
        ans -= prob * log(prob, base)
    return ans
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/%E7%86%B5.py)】测试

```python
>>> from code.dicision_tree import entropy
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> entropy(Y)  # H(D)
0.9709505944546686
```

#### 条件熵（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_conditional_extropy.py)】code.dicision_tree.conditional_entropy

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_conditional_extropy.py

import collections
from math import log

def conditional_entropy(x, y, base=2):
    """计算随机变量X给定的条件下随机变量Y的条件熵H(Y|X)"""
    freq_y_total = collections.defaultdict(collections.Counter)  # 统计随机变量X取得每一个取值时随机变量Y的频数
    freq_x = collections.Counter()  # 统计随机变量X每一个取值的频数
    for i in range(len(x)):
        freq_y_total[x[i]][y[i]] += 1
        freq_x[x[i]] += 1
    ans = 0
    for xi, freq_y_xi in freq_y_total.items():
        res = 0
        for freq in freq_y_xi.values():
            prob = freq / freq_x[xi]
            res -= prob * log(prob, base)
        ans += res * (freq_x[xi] / len(x))
    return ans
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/%E6%9D%A1%E4%BB%B6%E7%86%B5.py)】测试

```python
>>> from code.dicision_tree import conditional_entropy
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> conditional_entropy([X[i][0] for i in range(len(X))], Y)  # H(D|X=x_1)
0.8879430945988998
```

#### 信息增益（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_information_gain.py)】code.dicision_tree.information_gain

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_information_gain.py

from ._conditional_extropy import conditional_entropy  # code.dicision_tree.conditional_entropy
from ._entropy import entropy  # code.dicision_tree.entropy

def information_gain(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益"""
    return entropy(y, base=base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=base)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/%E4%BF%A1%E6%81%AF%E5%A2%9E%E7%9B%8A.py)】测试

```python
>>> from code.dicision_tree import information_gain
>>> from code.example import load_li_5_1
>>> X, Y = load_example()
>>> information_gain(X, Y, idx=0)  # g(D,A1)
0.08300749985576883
```

#### 信息增益比（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_information_gain_ratio.py)】code.dicision_tree.information_gain_ratio

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_information_gain_ratio.py

from ._entropy import entropy  # code.dicision_tree.entropy
from ._information_gain import information_gain  # code.dicision_tree.information_gain

def information_gain_ratio(x, y, idx, base=2):
    """计算特征A(第idx个特征)对训练数据集D(输入数据x,输出数据y)的信息增益比"""
    return information_gain(x, y, idx, base=base) / entropy([x[i][idx] for i in range(len(x))], base=base)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/%E4%BF%A1%E6%81%AF%E5%A2%9E%E7%9B%8A%E6%AF%94.py)】测试

```python
>>> from code.dicision_tree import information_gain_ratio
>>> from code.example import load_li_5_1
>>> X, Y = load_example()
>>> information_gain_ratio(X, Y, idx=0)  # gR(D,A1)
0.05237190142858302
>>> information_gain_ratio(X, Y, idx=1)  # gR(D,A2)
0.3524465495205019
>>> information_gain_ratio(X, Y, idx=2)  # gR(D,A3)
0.4325380677663126
>>> information_gain_ratio(X, Y, idx=3)  # gR(D,A4)
0.23185388128724224
```

## 5.3.1 决策树的生成-ID3算法

#### ID3算法生成决策树-不包含剪枝（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_id3_without_pruning.py)】code.dicision_tree.DecisionTreeID3WithoutPruning

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_id3_without_pruning.py

import collections
from ._conditional_extropy import conditional_entropy  # code.dicision_tree.conditional_entropy
from ._entropy import entropy  # code.dicision_tree.entropy

class DecisionTreeID3WithoutPruning:
    """ID3生成算法构造的决策树（仅支持离散型特征）-不包括剪枝"""

    class Node:
        def __init__(self, mark, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = collections.Counter(y)

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0])

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0])

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0])

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], use_feature=best_feature_idx)
        features = set()
        sub_x = collections.defaultdict(list)
        sub_y = collections.defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/ID3%E7%AE%97%E6%B3%95%E7%94%9F%E6%88%90%E5%86%B3%E7%AD%96%E6%A0%91(%E4%B8%8D%E5%8C%85%E5%90%AB%E5%89%AA%E6%9E%9D).py)】测试

```python
>>> from code.dicision_tree import DecisionTreeID3WithoutPruning
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> decision_tree = DecisionTreeID3WithoutPruning(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"])
>>> decision_tree
有自己的房子 = 是 -> 是
有自己的房子 = 否 :
  有工作 = 是 -> 是
  有工作 = 否 -> 否
```

## 5.3.2 决策树的生成-C4.5的生成算法

> 【补充说明】C4.5算法在生成的过程中，除了用信息增益比来选择特征外，还增加了通过动态定义将连续属性值分隔成一组离散间隔的离散属性，从而支持了连续属性的情况。**以下实现的内容为书中描述的C4.5生成算法！**

#### C4.5的生成算法生成决策树-不包含剪枝（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_c45_without_pruning.py)】code.dicision_tree.DecisionTreeC45WithoutPruning

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_c45_without_pruning.py

from ._decision_tree_id3_without_pruning import DecisionTreeID3WithoutPruning  # code.dicision_tree.DecisionTreeID3WithoutPruning
from ._entropy import entropy  # code.dicision_tree.entropy

class DecisionTreeC45WithoutPruning(DecisionTreeID3WithoutPruning):
    """C4.5生成算法构造的决策树（仅支持离散型特征）-不包含剪枝"""

    def information_gain(self, x, y, idx):
        """重写计算信息增益的方法，改为计算信息增益比"""
        return super().information_gain(x, y, idx) / entropy([x[i][idx] for i in range(len(x))], base=self.base)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/C4.5%E7%9A%84%E7%94%9F%E6%88%90%E7%AE%97%E6%B3%95(%E4%B8%8D%E5%8C%85%E6%8B%AC%E5%89%AA%E6%9E%9D).py)】测试

```python
>>> from code.dicision_tree import DecisionTreeC45WithoutPruning
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> decision_tree = DecisionTreeC45WithoutPruning(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"])
>>> decision_tree
有自己的房子 = 是 -> 是
有自己的房子 = 否 :
  有工作 = 是 -> 是
  有工作 = 否 -> 否
```

## 5.4 决策树的剪枝

#### 【问题】决策树的剪枝为什么可以使用动态规划？

这是树形DP的标准案例，即每一个结点在计算时，先计算出所有子结点的最优解，然后其根据子结点的最优解计算当前结点的最优解。

> 参考资料：[树形DP - OI Wiki](https://oi-wiki.org/dp/tree/)

#### ID3算法生成决策树-包含剪枝（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_id3.py)】code.decision_tree.DecisionTreeID3

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/dicision_tree/_decision_tree_id3.py

import collections
from ._conditional_extropy import conditional_entropy  # code.dicision_tree.conditional_entropy
from ._entropy import entropy  # code.dicision_tree.entropy

class DecisionTreeID3:
    """ID3生成算法构造的决策树（仅支持离散型特征）"""

    class Node:
        def __init__(self, mark, ee, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点
            self.ee = ee  # 以当前结点为叶结点的经验熵

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0, alpha=0.05):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值
        self.alpha = alpha  # 决策树剪枝的参数

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成
        self._pruning(self.root)  # 决策树剪枝

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = collections.Counter(y)
        ee = entropy(y, base=self.base)  # 计算以当前结点为叶结点的经验熵

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0], ee)

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], ee, use_feature=best_feature_idx)
        features = set()
        sub_x = collections.defaultdict(list)
        sub_y = collections.defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def _pruning(self, node):
        # 处理当前结点为叶结点的情况：不剪枝，直接返回
        if node.is_leaf:
            return 1, node.ee

        # 计算剪枝（以当前结点为叶结点）的损失函数
        loss1 = node.ee + 1 * self.alpha

        # 计算不剪枝的损失函数
        num, ee = 1, 0
        for child in node.children.values():
            child_num, child_ee = self._pruning(child)
            num += child_num
            ee += child_ee
        loss2 = ee + num * self.alpha

        # 处理需要剪枝的情况
        if loss1 < loss2:
            node.children = {}
            return 1, node.ee

        # 处理不需要剪枝的情况
        else:
            return num, ee

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/ID3%E7%AE%97%E6%B3%95%E7%94%9F%E6%88%90%E5%86%B3%E7%AD%96%E6%A0%91(%E5%8C%85%E5%90%AB%E5%89%AA%E6%9E%9D).py)】测试

```python
>>> from code.dicision_tree import DecisionTreeID3
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> DecisionTreeID3(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.2)
有自己的房子 = 是 -> 是
有自己的房子 = 否 :
  有工作 = 是 -> 是
  有工作 = 否 -> 否
>>> DecisionTreeID3(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"], alpha=0.3)
 -> 是
```

## 5.5 CART算法

#### 分类误差率

分类误差率，即分类错误的实例数占总实例数的比例。因为在叶结点中，我们选择实例数最大的类作为标记，所以不是该类的实例均会被标记错误。因此，分类误差率可定义为：
$$
error(p) = 1 - \max_k \ p_k, \hspace{1em} k=1,2,\cdots,K
$$

#### 【问题】为什么不用分类误差率衡量信息增益？

当某个结点中实例数最多的类，与其每个子结点中实例数最多的类均相同时；若用分类误差率衡量信息增益，因为不会考虑子结点中各类比例的变化，所以信息增益为0；但若用熵或基尼系数衡量信息增益，因为会考虑子结点中各类比例的变化，所以信息增益不为0。我们通过一个例子来看（以下“类标记”均表示以该结点为叶结点时的类标记，熵的单位均为比特）：

现有结点T0，其中包含A类实例80个，B类实例20个；当结点T0时，其类标记为A，分类误差率为0.2，熵为0.722。

现有一种分割方法，可以将结点T0分隔为结点T1和结点T2；其中结点T1包含A类实例60个，B类实例5个；结点T2包含A类实例20个，B类实例15个。此时结点T1的类标记为A，分类误差率为0.077，熵为0.391；结点T2的类标记为A，分类误差率为0.429，熵为0.985。对于结点T0，其分类误差率为$0.077×0.65+0.429×0.35=0.2$，其熵为$0.391×0.65+0.985×0.35=0.599$。

此时，使用分类误差率衡量的信息增益为0，使用熵衡量的信息增益为0.123。

#### CART分类树（Python+sklearn实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/CART%E5%88%86%E7%B1%BB%E6%A0%91-%E6%B5%8B%E8%AF%95%E5%AE%9E%E4%BE%8B1(sklearn%E5%AE%9E%E7%8E%B0).py)】测试实例1（例5.1的测试集）

```python
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.tree import export_text
>>> from code.example import load_li_5_1
>>> X, Y = load_li_5_1()
>>> N = len(X)
>>> n = len(X[0])

# 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
>>> y_list = list(set(Y))
>>> y_mapping = {c: i for i, c in enumerate(y_list)}
>>> x_list = [list(set(X[i][j] for i in range(N))) for j in range(n)]
>>> x_mapping = [{c: i for i, c in enumerate(x_list[j])} for j in range(n)]
>>> for i in range(N):
...     for j in range(n):
...         X[i][j] = x_mapping[j][X[i][j]]
>>> for i in range(N):
...     Y[i] = y_mapping[Y[i]]

>>> clf = DecisionTreeClassifier()
>>> clf.fit(X, Y)
>>> export_text(clf, feature_names=["年龄", "有工作", "有自己的房子", "信贷情况"],show_weights=True)
|--- 有自己的房子 <= 0.50
|   |--- 有工作 <= 0.50
|   |   |--- weights: [6.00, 0.00] class: 0
|   |--- 有工作 >  0.50
|   |   |--- weights: [0.00, 3.00] class: 1
|--- 有自己的房子 >  0.50
|   |--- weights: [0.00, 6.00] class: 1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/CART%E5%88%86%E7%B1%BB%E6%A0%91-%E6%B5%8B%E8%AF%95%E5%AE%9E%E4%BE%8B2(sklearn%E5%AE%9E%E7%8E%B0).py)】测试示例2（鸢尾花数据集）

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.tree import export_text

>>> iris = load_iris()
>>> X = iris.data
>>> Y = iris.target
>>> x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
>>> clf = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0)
>>> clf.fit(x1, y1)
>>> export_text(clf, feature_names=iris.feature_names, show_weights=True)
|--- petal width (cm) <= 0.75
|   |--- weights: [34.00, 0.00, 0.00] class: 0
|--- petal width (cm) >  0.75
|   |--- petal length (cm) <= 4.95
|   |   |--- petal width (cm) <= 1.65
|   |   |   |--- weights: [0.00, 29.00, 0.00] class: 1
|   |   |--- petal width (cm) >  1.65
|   |   |   |--- weights: [0.00, 1.00, 3.00] class: 2
|   |--- petal length (cm) >  4.95
|   |   |--- weights: [0.00, 1.00, 32.00] class: 2
>>> clf.score(x2, y2)
0.98
```

#### CART回归树（Python+sklearn实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC5%E7%AB%A0_%E5%86%B3%E7%AD%96%E6%A0%91/CART%E5%9B%9E%E5%BD%92%E6%A0%91-%E6%B5%8B%E8%AF%95%E5%AE%9E%E4%BE%8B(sklearn%E5%AE%9E%E7%8E%B0).py)】测试示例（波士顿房价数据集）

```python
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.tree import DecisionTreeRegressor
>>> from sklearn.tree import export_text
>>> boston = load_boston()
>>> X = boston.data
>>> Y = boston.target
>>> x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
>>> clf = DecisionTreeRegressor(ccp_alpha=0.16, random_state=0)
>>> clf.fit(x1, y1)
>>> export_text(clf, feature_names=list(boston.feature_names))
|--- LSTAT <= 7.88
|   |--- RM <= 7.43
......
|   |   |   |   |--- NOX >  0.68
|   |   |   |   |   |--- value: [9.21]
>>> clf.score(x2, y2)  # 平方误差
0.7217463605968275
```

