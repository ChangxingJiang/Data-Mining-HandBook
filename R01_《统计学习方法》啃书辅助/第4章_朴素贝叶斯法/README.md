# 《统计学习方法》啃书辅助：第 4 章 朴素贝叶斯法

**朴素贝叶斯法对数据的要求**：数据满足特征条件独立假设，即用于分类的特征在类确定的条件下都是条件独立的。

**朴素贝叶斯法的学习过程**：基于特征条件独立假设学习输入输出的联合概率分布。即通过先验概率分布 $P(Y=c_k)$ 和条件概率分布 $P(X^{(j)}=x^{(j)}|Y=c_k)$ 实现对联合概率分布 $P(X,Y)$ 的估计。

**朴素贝叶斯法的预测过程**：基于模型，对给定的输入 x，利用贝叶斯定理求出后验概率最大的输出 y。

**朴素贝叶斯法的类别划分**：

- 用于解决分类问题的监督学习模型
- 概率模型：模型取条件概率分布形式 $P(y|x)$
- 参数化模型：假设模型参数的维度固定
- 生成模型：由数据直接学习联合概率分布 $P(X,Y)$，然后求出概条件概率分布 $P(Y|X)$

**朴素贝叶斯法的主要优点**：学习与预测的效率很高，易于实现。

**朴素贝叶斯法的主要缺点**：因为特征条件独立假设很强，分类的性能不一定很高。

> 【扩展阅读】[sklearn 中文文档：1.9 朴素贝叶斯](https://sklearn.apachecn.org/docs/master/10.html)

---

【补充说明】书中介绍的朴素贝叶斯法只适用于离散型特征，如果是连续型特征，还需要考虑连续型特征的概率密度函数。详见“延伸阅读”。

【补充说明】这里的“特征条件独立假设”即下文中“朴素贝叶斯法对条件概率分布作出的条件独立性假设”。

#### 贝叶斯定理

首先根据条件概率的定义，可得如下定理。

> **【定理 1】乘法定理 （来自浙江大学《概率论与数理统计》第四版 P. 16）**
>
> 设 $P(A)>0$，则有
>
> $$
> P(AB) = P(B|A) P(A)
> $$

接着给出划分的定义。

> **【定义 2】划分 （来自浙江大学《概率论与数理统计》第四版 P. 17）**
>
> 设 S 为试验 E 的样本空间，$B_1,B_2,\cdots,B_n$ 为 E 的一组事件。若
>
> 1. $B_i B_j = \varnothing$，$i \ne j$，$i,j=1,2,\cdots,n$；
> 2. $B_1 \cup B_2 \cup \cdots \cup B_n = S$，
>
> 则称 $B_1,B_2,\cdots,B_n$ 为样本空间 S 的一个划分。

根据划分的定义，我们知道若 $B_1,B_2,\cdots,B_n$ 为样本空间的一个划分，那么，对每次试验，事件 $B_1,B_2,\cdots,B_n$ 中必有一个且仅有一个发生。于是得到全概率公式和贝叶斯公式。

> **【定理 3】全概率公式 （来自浙江大学《概率论与数理统计》第四版 P. 18）**
>
> 设试验 E 的样本空间为 S，A 为 E 的事件，$B_1,B_2,\cdots,B_n$ 为 S 的一个划分，且 $P(B_i)>0 \ (i=1,2\cdots,n)$，则
>
> $$
> P(A) = P(A|B_1) P(B_1) + P(A|B_2) P(B_2) + \cdots + P(A|B_n) P(B_n)
> $$

> **【定理 4】贝叶斯公式 （来自浙江大学《概率论与数理统计》第四版 P. 18）**
>
> 设试验 E 的样本空间为 S，A 为 E 的事件，$B_1,B_2,\cdots,B_n$ 为 S 的一个划分，且 $P(A)>0$，$P(B_i)>0 \ (i=1,2\cdots,n)$，则
>
> $$
> P(B_i|A) = \frac{P(A|B_i) P(B_i)}{\sum_{j=1}^n P(A|B_j) P(B_j)}, \ i=1,2,\cdots,n
> $$

## 4.1.1 朴素贝叶斯法的学习与分类-基本方法

【补充说明】作出特征条件独立假设，即条件独立性的假设后，参数规模降为 $K \sum_{j=1}^n S_j$。

【补充说明】$argmax$ 函数用于计算因变量取得最大值时对应的自变量的点集。求函数 $f(x)$ 取得最大值时对应的自变量 $x$ 的点集可以写作

$$
arg \max_{x} f(x)
$$

【补充说明】在公式 4.4、4.5、4.6 中，等号右侧求和符号内的 k 并不影响求和符号外的 k，但是相同符号会带来些许歧义，或许可以将分母中的 k 改写为其他字母，例如可以将公式 4.4 改写为下式：

$$
P(Y=c_k|X=x)=\frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_i P(X=x|Y=c_i)P(Y=c_i)}
$$

## 4.1.2 朴素贝叶斯法后验概率最大化的含义

【补充说明】“期望风险”的前置知识在“1.3.2 策略”。

#### 期望的下标符号的含义

期望的下标符号似乎没有确切的定义。在[【stackexchange 的问题：Subscript notation in expectations】](https://stats.stackexchange.com/questions/72613/subscript-notation-in-expectations)中，提及使用了期望的下标符号的[【wikipedia 词条：Law of total expectation】](https://en.wikipedia.org/wiki/Law_of_total_expectation)现在也已经没有期望的下标符号了。在[@Alecos Papadopoulos](https://stats.stackexchange.com/users/28746/alecos-papadopoulos)的回答中，提到有两种可能：

第一种是将下标符号中的变量作为条件，即

$$
E_X [L(Y,f(X))] = E[L(Y,f(X))|X]
$$

第二种是将下标符号中的变量用作计算平均，即

$$
E_X [L(Y,f(X))] = \sum_{x \in X} L(Y,f(X)) P(X=x)
$$

显然，书中所使用的期望的下标符号是第二种含义。于是下式

$$
R_{exp}(f)=E_X \sum_{k=1}^K [L(c_k,f(X))]P(c_k|X)
$$

可以理解为

$$
R_{exp}(f)=E \sum_{x \in X} \bigg{[} \sum_{k=1}^K [L(c_k,f(X))]P(c_k|X=x) \bigg{]} P(X=x)
$$

#### 根据期望风险最小化准则推导后验概率最大化准则 （不使用期望的下标符号）

设 X 为 n 维向量的集合，类标记集合 Y 有 k 个值。已知期望是对联合分布 $P(X,Y)$ 取的，所以期望风险函数可以写作

$$
\begin{align}
R_{exp}(f)
& = E[L(Y,f(X))] \\
& = L(Y,f(X)) P(X,Y) \\
& = \sum_{i=1}^n \sum_{j=1}^k L(Y_j,f(X_i)) P(X=X_i,Y=Y_j) \\
& = \sum_{i=1}^n \Big{[} \sum_{j=1}^k L(Y_j,f(X_i)) P(Y=Y_j|X=X_i) \Big{]} P(X=X_i) \\
& = \sum_{i=1}^n \Big{[} \sum_{j=1}^k P(Y \ne Y_j|X=X_i) \Big{]} P(X=X_i)
\end{align}
$$

因为特征条件独立假设，所以为了使期望风险最小化，只需对 $X=x$ 逐个极小化，后续证明与书中相同，不再赘述。

## 4.2 朴素贝叶斯法的参数估计

#### 极大似然估计

极大似然估计，也称最大似然估计，其核心思想是：在已经得到试验结果的情况下，我们应该寻找使这个结果出现的可能性最大的参数值作为参数的估计值。

#### 朴素贝叶斯算法（原生 Python 实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_hashmap.py)】code.naive_bayes.NaiveBayesAlgorithmHashmap（哈希表存储先验概率和条件概率）

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_hashmap.py

import collections

class NaiveBayesAlgorithmHashmap:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数
        self.n = len(x[0])  # 维度数

        count1 = collections.Counter(y)  # 先验概率的分子，条件概率的分母
        count2 = [collections.Counter() for _ in range(self.n)]  # 条件概率的分子
        for i in range(self.N):
            for j in range(self.n):
                count2[j][(x[i][j], y[i])] += 1

        # 计算先验概率和条件概率
        self.prior = {k: v / self.N for k, v in count1.items()}
        self.conditional = [{k: v / count1[k[1]] for k, v in count2[j].items()} for j in range(self.n)]

    def predict(self, x):
        best_y, best_score = 0, 0
        for y in self.prior:
            score = self.prior[y]
            for j in range(self.n):
                score *= self.conditional[j][(x[j], y)]
            if score > best_score:
                best_y, best_score = y, score
        return best_y
```

二维数组存储先验概率和条件概率。

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_array.py)】code.naive_bayes.NaiveBayesAlgorithmArray

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_array.py

class NaiveBayesAlgorithmArray:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        table1 = [0] * self.K
        for i in range(self.N):
            table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = table1[k] / self.N

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = table2[j][k][t] / table1[k]

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= 0
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC4%E7%AB%A0_%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B3%95/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%AE%97%E6%B3%95.py)】测试

```python
>>> from code.naive_bayes import NaiveBayesAlgorithmArray
>>> from code.naive_bayes import NaiveBayesAlgorithmHashmap
>>> dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
...             (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
...             (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
...            [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
>>> naive_bayes_1 = NaiveBayesAlgorithmHashmap(*dataset)
>>> naive_bayes_1.predict([2, "S"])
-1
>>> naive_bayes_2 = NaiveBayesAlgorithmArray(*dataset)
>>> naive_bayes_2.predict([2, "S"])
-1
```

#### 贝叶斯估计（原生 Python 实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_with_smoothing.py)】code.naive_bayes.NaiveBayesAlgorithmWithSmoothing

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_with_smoothing.py

class NaiveBayesAlgorithmWithSmoothing:
    """贝叶斯估计（仅支持离散型数据）"""

    def __init__(self, x, y, l=1):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数
        self.l = l  # 贝叶斯估计的lambda参数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        self.table1 = [0] * self.K
        for i in range(self.N):
            self.table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        self.table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                self.table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = (self.table1[k] + self.l) / (self.N + self.l * self.K)

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = (self.table2[j][k][t] + self.l) / (self.table1[k] + self.l * self.Sj[j])

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= self.l / (self.table1[k] + self.l * self.Sj[j])
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/naive_bayes/_naive_bayes_algorithm_with_smoothing.py)】测试

```python
>>> from code.naive_bayes import NaiveBayesAlgorithmWithSmoothing
>>> dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
>>> naive_bayes = NaiveBayesAlgorithmWithSmoothing(*dataset)
>>> naive_bayes.predict([2, "S"])
-1
```

## 延伸阅读

以上的朴素贝叶斯法只适用于离散型特征，如果是连续型特征，还需要考虑连续型特征的概率密度函数。当假设连续型特征服从不同的参数时，有不同的方法：

- 高斯朴素贝叶斯（GNB）：假设连续型特征服从高斯分布（正态分布），使用极大似然估计求分布参数。
- 多项分布朴素贝叶斯（MNB）：假设连续型特征服从多项式分布，使用个极大似然估计求模型参数。
- 补充朴素贝叶斯（CNB）：假设连续型特征服从多项式分布，使用来自每个类的补数的统计数据来计算模型的权重，是标准多项式朴素贝叶斯（MNB）的一种改进，特别适用于不平衡数据集。
- 伯努利朴素贝叶斯：假设连续型特征服从多重伯努利分布。

#### 支持连续型特征的朴素贝叶斯（sklearn 实现）

【[源码地址](<https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC4%E7%AB%A0_%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%B3%95/%E6%94%AF%E6%8C%81%E8%BF%9E%E7%BB%AD%E5%9E%8B%E7%89%B9%E5%BE%81%E7%9A%84%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF(sklearn%E5%AE%9E%E7%8E%B0).py>)】测试

```python
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.naive_bayes import ComplementNB
>>> from sklearn.naive_bayes import BernoulliNB

>>> X, Y = load_breast_cancer(return_X_y=True)
>>> x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

>>> # 高斯朴素贝叶斯
>>> gnb = GaussianNB()
>>> gnb.fit(x1, y1)
>>> gnb.score(x2, y2)
0.9210526315789473

>>> # 多项分布朴素贝叶斯
>>> mnb = MultinomialNB()
>>> mnb.fit(x1, y1)
>>> mnb.score(x2, y2)
0.9105263157894737

>>> # 补充朴素贝叶斯
>>> mnb = ComplementNB()
>>> mnb.fit(x1, y1)
>>> mnb.score(x2, y2)
0.9052631578947369

>>> # 伯努利朴素贝叶斯
>>> bnb = BernoulliNB()
>>> bnb.fit(x1, y1)
>>> bnb.score(x2, y2)
0.6421052631578947
```
