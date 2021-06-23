# 《统计学习方法》啃书辅助：第6章 逻辑斯谛回归与最大熵模型

**逻辑斯谛回归模型的学习过程**：通过求解对数似然函数极大化，得到条件概率分布 $P(Y|X)$。

**逻辑斯谛回归模型的预测过程**：直接使用模型的条件概率分布 $P(Y|X)$ 进行预测。

**逻辑斯谛回归模型的类别划分**：

- 用于解决分类问题的监督学习模型
- 概率模型&非概率模型：既可以看作是概率模型，也可以看作是非概率模型
- 参数化模型：假设模型参数的维度固定
- 判别模型：由数据直接学习决策函数 $f(Y|X)$

**逻辑斯谛回归模型的主要优点**：直接对分类进行建模，无需事先假设数据分布；在预测类别的同时，还可以得到近似概率预测

**逻辑斯谛回归模型的主要缺点**：不太适合特征空间很大的情况。

---

**最大熵模型的学习过程**：通过求解对数似然函数极大化或对偶函数极大化，得到条件概率分布 $P(Y|X)$。

**最大熵模型的预测过程**：直接使用模型的条件概率分布 $P(Y|X)$ 进行预测。

**最大熵模型的类别划分**：

- 用于解决分类问题的监督学习模型
- 判别模型：由数据直接学习决策函数 $f(Y|X)$

**最大熵模型的主要优点**：可以灵活地选择特征，使用不同类型的特征。

**最大熵模型的主要缺点**：计算量巨大。

---

【补充说明】逻辑斯谛回归模型，也称逻辑回归模型、LR 模型。

【延伸阅读】[改进的迭代尺度法(IIS)详细解析 | 统计学习方法学习笔记 | 数据分析 | 机器学习 - 舟晓南的文章 - 知乎](https://zhuanlan.zhihu.com/p/234553402)

## 6.1 逻辑斯谛回归模型

> **【重点】逻辑斯谛回归模型的含义（来自《统计学习方法》P.93）**
>
> 在逻辑斯蒂回归模型中，输出 $Y=1$ 的对数几率是输入 x 的线性函数。或者说，输出 $Y=1$ 的对数几率是由输入 x 的线性函数表示的模型，即逻辑斯蒂回归模型。

#### Sigmoid 函数的性质

$$
f(x) = \frac{1}{1+e^{-x}}
$$

其性质如下：

- 定义域为 $O(-\infty,+\infty)$，值域为 $(0,1)$，在定义域内连续，严格单调递增；
- 以 $(0,\frac{1}{2})$ 中心对称；
- 其一阶导数为 $f'(x) = f(x) (1-f(x))$；
- 具有简单变形 $f(x)=\frac{e^x}{1+e^x}$（分子分母同乘 $e^x$）。

#### 为什么要用 sigmoid 函数？

【延伸阅读】[通过逻辑回归的 sigmoid 函数把线性回归转化到 [0, 1] 之间，这个值为什么可以代表概率？ - 盛傢伟的回答 - 知乎](https://www.zhihu.com/question/41647192/answer/1823634532)

> （结合个人理解，如有错误，还望指出）

对于二类分类问题，有输入变量 $x \in R^n$ 和输出变量 $y \in \{0,1\}$。其基本的线性回归的形式为：

$$
y = w^T x + b
$$

其中 $w \in R^n$ 是权值向量，$b \in R$ 是偏置。显然，线性回归产生的预测值是一个连续变量。

很自然地，我们首先考虑通过线性回归直接预测正类的概率，即 $P(Y=1|x)$。但是线性回归的预测值的值域是 $(-\infty,+\infty)$，而我们希望这个预测值的值域是 $(0,1)$，于是我们需要修改线性回归预测的内容。

首先，我们考虑将线性回归改为预测事件的几率，即该事件发生的概率与该事件不发生的概率的比值，即 $\frac{P(Y=1|x)}{1-P(Y=1|x)}$。此时，我们希望预测值的值域变为 $(0,+\infty)$。

考虑到在进行分类时，我们是通过比较两个条件概率值的大小决定分类的；因此，在线性回归外套一个单调递增函数并不会影响分类结果；同时，为便于计算，我们还希望外套的这个函数任意阶可导。于是，可以选择外套底数大于 1 的指数函数，不妨先取底数为 $e$：

$$
\frac{P(Y=1|x)}{1-P(Y=1|x)} = e^{w^T x + b}
$$

此时，线性回归拟合的就是事件对数几率，即

$$
log \frac{P(Y=1|x)}{1-P(Y=1|x)} = w^T x + b
$$

综上所述，以上过程就是在寻找比正类概率更适合线性回归拟合的目标，而找到的结果，就是逻辑斯谛回归模型。

#### 似然函数

【扩展阅读】[如何理解似然函数? - Yeung Evan 的回答 - 知乎](https://www.zhihu.com/question/54082000/answer/145495695)

> **【如何理解似然函数? - Yeung Evan 的回答 - 知乎】摘要**
>
> 似然函数定义为给定样本值 $x$ 关于未知参数 $\theta$ 的函数，是样本集中各个样本的联合概率：
>
> $$
> L(\theta|x) = f(x|\theta)
> $$
>
> 其中，$x$ 是输入变量 $X$ 取到的值，即 $X=x$；$\theta$ 为未知参数，$f(x|\theta)$ 是给定 $\theta$ 的情况下 $x$ 的联合密度函数。
>
> 似然函数 $L(\theta|x)$ 是关于 $\theta$ 的函数，密度函数 $f(x|\theta)$ 是关于 $x$ 的函数，两者仅在函数值上相等，而定义域和对应关系并不相等。
>
> 似然函数 $L(\theta|x)$ 表示给定样本 $X=x$ 下参数 $\theta$ 为真实值的可能性；密度函数 $f(x|\theta)$ 表示给定 $\theta$ 下样本随机向量 $X=x$ 的可能性。

## 6.2 最大熵模型

> **【重点】最大熵原理（来自《统计学习方法》P.95）**
>
> 直观地，最大熵原理认为要选择的概率模型必须满足已有的事实，即约束条件。在没有更多信息的情况下，那么不确定的部分都是“等可能的”。最大熵原理通过熵的最大化来表示等可能性。“等可能”不容易操作，而熵则是一个可优化的数值指标。

#### 单纯形

单纯形是三角形和四面体的一种泛化，即凸多面体；$k$ 维单纯形是指包含 $(k+1)$ 个结点的凸多面体。

#### 经验分布

当分布由样本数据估计得到时，所对应的联合分布 $P(X,Y)$ 和边缘分布 $P(X)$ 就是联合分布 $P(X,Y)$ 的经验分布和边缘分布 $P(X)$ 的经验分布。

#### 约束条件的含义是什么？

对于约束条件公式：

$$
\sum_{x,y} \tilde{P}(x) P(y|x) f(x,y) = \sum_{x,y} \tilde{P}(x,y) f(x,y)
$$

我们可以将其理解为，模型必须完全服从所有 $f(x,y)=1$ 的实例点的概率总和，但对所有 $f(x,y)=1$ 的实例点之间不做区分。

【扩展阅读】[如何理解最大熵模型里面的特征？ - SleepyBag 的回答 - 知乎](https://www.zhihu.com/question/24094554/answer/1507080982)

#### 最大熵模型为什么可以通过对偶问题求解？

关于拉格朗日对偶性，有如下定理：

> **《统计学习方法》定理 C.2 (P. 450)**
>
> 考虑原始问题和对偶问题。假设函数 $f(x)$ 和 $c_i(x)$ 是凸函数，$h_j(x)$ 是仿射函数；并且假设不等式约束 $c_i(x)$ 是严格可行的，即存在 x，对所有 i 有 $c_i(x)<0$，则存在 $x^*$，$\alpha^*$，$\beta^*$，使 $x^*$ 是原始问题的解，$\alpha^*$，$\beta^*$ 是对偶问题的解，并且
>
> $$
> p^* = d^* = L(x^*,\alpha^*,\beta^*)p^* = d^* = L(x^*,\alpha^*,\beta^*)
> $$

根据上述定理，最大熵模型可以通过求解对偶问题来求解原始问题的条件如下：

1. $-H(P)=\sum_{x,y} \tilde{P}(x) P(y|x) log P(y|x)$ 是凸函数；
2. $E_p(f_i) - E_{\tilde{P}}(f_i) = 0$ 是仿射函数；
3. $\sum_y P(y|x) = 1$ 是仿射函数。

第 3 个条件显然成立，下面依次证明第 1 个条件和第 2 个条件。

在最大熵模型中，分类模型是条件概率分布 $P(Y|X)$，学习的目标是用最大熵原理选择最好的分类模型，因此自变量就是条件概率分布 $P(Y|X)$。根据训练数据集确定的联合分布的经验分布 $\tilde{P}(X,Y)$ 和边缘分布的经验分布 $\tilde{P}(X)$ 均应视为常量。

##### 证明第 1 个条件

求 $-H(P)$ 对 $P(y|x)$ 的二阶偏导数

$$
\frac{\partial (-H(P))}{\partial P(y|x)} = \sum_{x,y} \frac{\tilde{P}(x)}{P(y|x)}
$$

因为有 $\tilde{P}(x) = \frac{v(X=x)}{N} >= 0$，$P(y|x) >= 0$，所以二阶偏导数恒非负。于是，第 1 个条件得证。

##### 证明第 2 个条件

将 $E_p(f_i)$ 和 $E_{\tilde{P}}(f_i)$ 展开，于是有

$$
\begin{align}
E_p(f_i) - E_{\tilde{P}}(f_i)
& = \sum_{x,y} \tilde{P}(x) P(y|x) f(x,y) - \sum_{x,y} \tilde{P}(x,y) f(x,y) \\
& = \sum_{x,y} f(x,y) \big[ \tilde{P}(x) P(y|x) - \tilde{P}(x,y) \big]
\end{align}
$$

于是，$E_p(f_i) - E_{\tilde{P}}(f_i)$ 显然是关于 $P(Y|X)$ 的仿射函数。

#### 最大熵模型对数似然函数中指数的含义？

因为在训练数据集中，$(x,y)$ 的样本可能不止一个，而连乘符号中仅区分了不同的 $(x,y)$。如果没有指数的话，那么无论 $(x,y)$ 的出现频数是多少，均只乘了一次。

$\tilde{P}(x,y)$ 正是 $(x,y)$ 的样本的频数，因此使用 $\tilde{P}(x,y)$ 作为指数，以表示 $(x,y)$ 的样本需要连乘的次数。

## 6.3 模型学习的最优化算法

#### Jensen 不等式（琴生不等式）

$$
f(\sum_{i=1}^n \lambda_i x_i) \le \sum_{i=1}^n \lambda_i f(x_i)
$$

在这里的使用中，将 $\sum_{i=1}^n \frac{f_i(x,y)}{f^\#(x,y)}=1$ 视作常量。

#### 一维牛顿法求 $f(x)=0$ 的值（Python 实现）

```python
from scipy.misc import derivative

def newton_method_linear(func, args=(), error=1e-6, dx=1e-6):
    """一维牛顿法求f(x)=0的值

    :param func: 目标函数
    :param args: 参数列表
    :param error: 容差
    :param dx: 计算导数时使用的dx
    :return:
    """
    x0, y0 = 0, func(0, *args)
    while True:
        d = derivative(func, x0, args=args, dx=dx)  # 计算一阶导数
        x1 = x0 - y0 / d
        if abs(x1 - x0) < error:
            return x1
        x0, y0 = x1, func(x1, *args)
```

```python
>>> def f(x, k):
...     return (x - k) ** 3
>>> newton_method_linear(f, args=(2,))
1.999998535982025
```

#### 改进的迭代尺度算法 IIS（Python 实现）

```python
from math import e
from ._newton_method_linear import newton_method_linear

def improved_iterative_scaling(x, y, features, error=1e-6):
    """改进的迭代尺度法求最大熵模型

    :param x: 输入变量
    :param y: 输出变量
    :param features: 特征函数列表
    :return:
    """
    n_samples = len(x)  # 样本数量
    n_features = len(features)  # 特征函数数量

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 不同的x的数量
    n_y = len(y_list)  # 不同的y的数量
    n_total = n_x * n_y  # 不同样本的总数

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    print("边缘分布的经验分布", d2)

    # 计算特征函数关于经验分布的期望值:EP(fi) (empirical_joint_distribution_each_feature)
    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    d3 = [0.0] * n_features  # empirical_joint_distribution_each_feature
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features
    for j in range(n_features):
        for xi in range(n_x):
            for yi in range(n_y):
                if features[j](list(x_list[xi]), y_list[yi]):
                    d3[j] += d1[xi][yi]
                    nn[xi][yi] += 1

    print("特征函数关于经验分布的期望值:", d3)
    print("所有特征在(x,y)出现的次数:", nn)

    # 定义w的初值和模型P(Y|X)的初值
    w0 = [0] * n_features  # w的初值：wi=0
    p0 = [[1 / n_total] * n_y for _ in range(n_x)]  # 当wi=0时，P(Y|X)的值

    change = True
    while change:
        change = False

        # 遍历各个特征条件以更新w
        for j in range(n_features):
            def func(d, jj):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        if features[j](list(x_list[xxi]), y_list[yyi]):
                            res += d2[xxi] * p0[xxi][yyi] * pow(e, d * nn[xxi][yyi])
                res -= d3[jj]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(j,))

            # 更新wi的值
            w0[j] += dj
            if abs(dj) >= error:
                change = True

        # 计算新的模型
        p1 = [[0.0] * n_y for _ in range(n_x)]
        for xi in range(n_x):
            for yi in range(n_y):
                for j in range(n_features):
                    if features[j](list(x_list[xi]), y_list[yi]):
                        p1[xi][yi] += w0[j]
                p1[xi][yi] = pow(e, p1[xi][yi])
            total = sum(p1[xi][yi] for yi in range(n_y))
            if total > 0:
                for yi in range(n_y):
                    p1[xi][yi] /= total

        if not change:
            ans = {}
            for xi in range(n_x):
                for yi in range(n_y):
                    ans[(tuple(x_list[xi]), y_list[yi])] = p1[xi][yi]
            return w0, ans

        p0 = p1
```

```python
>>> from code.maximum_entropy_model import improved_iterative_scaling
>>> dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]
>>> def f1(xx, yy):
...     return xx == [1] and yy == 1
>>> def f2(xx, yy):
...     return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)
>>> improved_iterative_scaling(dataset[0], dataset[1], [f1])
[-0.40546481008458535],
{((1,), 1): 0.2500000558794252, ((1,), 2): 0.3749999720602874, ((1,), 3): 0.3749999720602874, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
>>> improved_iterative_scaling(dataset[0], dataset[1], [f2])
[0.40546793651918606],
{((1,), 1): 0.24999946967330844, ((1,), 2): 0.3750002651633458, ((1,), 3): 0.3750002651633458, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
```

#### 最大熵模型学习的 BFGS 算法

```python
from math import e
from math import log

import numpy as np

from ..gradient_descent import golden_section_for_line_search
from ..gradient_descent import partial_derivative


def bfgs_algorithm_for_maximum_entropy_model(x, y, features, error=1e-6, distance=20, maximum=1000):
    """最大熵模型学习的BFGS算法

    :param x: 输入变量
    :param y: 输出变量
    :param features: 特征函数列表
    :param error: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    n_samples = len(x)  # 样本数量
    n_features = len(features)  # 特征函数数量

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 不同的x的数量
    n_y = len(y_list)  # 不同的y的数量
    n_total = n_x * n_y  # 不同样本的总数

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    print("边缘分布的经验分布", d2)

    # 计算特征函数关于经验分布的期望值:EP(fi) (empirical_joint_distribution_each_feature)
    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    d3 = [0.0] * n_features  # empirical_joint_distribution_each_feature
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features
    for j in range(n_features):
        for xi in range(n_x):
            for yi in range(n_y):
                if features[j](list(x_list[xi]), y_list[yi]):
                    d3[j] += d1[xi][yi]
                    nn[xi][yi] += 1

    print("特征函数关于经验分布的期望值:", d3)
    print("所有特征在(x,y)出现的次数:", nn)

    def func(ww):
        """目标函数"""
        res = 0
        for xxi in range(n_x):
            t1 = 0
            for yyi in range(n_y):
                t2 = 0
                for jj in range(n_features):
                    if features[jj](list(x_list[xxi]), y_list[yyi]):
                        t2 += ww[jj]
                t1 += pow(e, t2)
            res += d2[xxi] * log(t1, e)

        for xxi in range(n_x):
            for yyi in range(n_y):
                t3 = 0
                for jj in range(n_features):
                    if features[jj](list(x_list[xxi]), y_list[yyi]):
                        t3 += ww[jj]
                res -= d1[xxi][yyi] * t3

        return res

    # 定义w的初值和B0的初值
    w0 = [0] * n_features  # w的初值：wi=0
    B0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)

    for k in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, w0)

        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < error:
            break

        # 计算pk
        if k == 0:
            pk = - B0 * g0  # 若numpy计算逆矩阵时有0，则对应位置会变为inf
        else:
            pk = - (B0 ** -1) * g0

        # 一维搜索求lambda_k
        def f(xx):
            """pk 方向的一维函数"""
            x2 = [w0[jj] + xx * float(pk[jj][0]) for jj in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # print(k, "lk:", lk)

        # 更新当前点坐标
        w1 = [w0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # print(k, "w1:", w1)

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        # 计算新的模型

        nabla = partial_derivative(func, w1)

        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < error:
            w0 = w1
            break

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        B1 = B0 + (yk * yk.T) / (yk.T * dk) + (B0 * dk * dk.T * B0) / (dk.T * B0 * dk)

        B0 = B1
        w0 = w1

    p1 = [[0.0] * n_y for _ in range(n_x)]
    for xi in range(n_x):
        for yi in range(n_y):
            for j in range(n_features):
                if features[j](list(x_list[xi]), y_list[yi]):
                    p1[xi][yi] += w0[j]
            p1[xi][yi] = pow(e, p1[xi][yi])
        total = sum(p1[xi][yi] for yi in range(n_y))
        if total > 0:
            for yi in range(n_y):
                p1[xi][yi] /= total

    ans = {}
    for xi in range(n_x):
        for yi in range(n_y):
            ans[(tuple(x_list[xi]), y_list[yi])] = p1[xi][yi]
    return w0, ans
```

```python
>>> from code.maximum_entropy_model import bfgs_algorithm_for_maximum_entropy_model
>>> dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]
>>> def f1(xx, yy):
...     return xx == [1] and yy == 1
>>> def f2(xx, yy):
...     return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)
>>> bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f1])
[-0.40546491364767995],
{((1,), 1): 0.2500000364613426, ((1,), 2): 0.3749999817693287, ((1,), 3): 0.3749999817693287, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
>>> bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f2])
[0.4054649114869288]
{((1,), 1): 0.2500000368664835, ((1,), 2): 0.3749999815667583, ((1,), 3): 0.3749999815667583, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
```
