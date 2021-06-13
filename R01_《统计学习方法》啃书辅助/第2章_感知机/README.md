# 《统计学习方法》啃书辅助：第 2 章 感知机

**感知机模型对数据的要求**：训练数据集中需存在某个超平面能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即训练数据集是线性可分的。因为只有当训练数据集是线性可分时，感知机学习算法才是收敛的；如果训练数据集线性不可分，则感知机学习算法不收敛，迭代结果会发生震荡。当训练数据集线性不可分时，可以使用线性支持向量机。

**感知机模型的学习过程**：依据训练数据集求得感知机模型，即求得模型参数 $w$ 和 $b$。

**感知机模型的预测过程**：通过学习得到的感知机模型，计算新的输入实例所对应的输出类别。

**感知机模型的类别划分**：

- 用于解决二类分类问题的监督学习模型
- 非概率模型：模型取函数形式（而非概率模型的条件概率分布形式）
- 线性模型：模型函数为线性函数
- 参数化模型：模型参数的维度固定
- 判别模型：直接学习决策函数 $f(x)$

**感知机模型的主要优点**：算法简单，易于实现。

**感知机模型的主要缺点**：要求训练数据集线性可分。

---

## 2.2 感知机学习策略

#### 常见范数的定义和性质

**范数** ，对应闵可夫斯基距离 (Minkowski distance) 。假设 n 维向量 $x = (x_1,x_2,\cdots,x_n)^T$，其 Lp 范数记作 $||x||_p$，定义为 $||x||_p = (|x_1|^p+|x_2|^p+\cdots+|x_n|^p)^{\frac{1}{p}}$ 。范数具有如下定义：

- 正定性：$||x|| \ge 0$，且有 $||x||=0 \Leftrightarrow x=0$；
- 正齐次性：$||cx|| = |c| \ ||x||$；
- 次可加性（三角不等式）：$||x+y|| \le ||x|| + ||y||$。

**L0 范数**

假设 n 维向量 $x = (x_1,x_2,\cdots,x_n)^T$，其 L0 范数记作 $||x||_0$，定义为向量中非 0 元素的个数。

**L1 范数**

假设 n 维向量 $x = (x_1,x_2,\cdots,x_n)^T$，其 L1 范数记作 $||x||_1$，定义为 $||x||_1 = |x_1|+|x_2|+\cdots+|x_n|$ 。向量的 L1 范数即为向量中各个元素绝对值之和，对应曼哈顿距离 (Manhattan distance)。

**L2 范数**

假设 n 维向量 $x = (x_1,x_2,\cdots,x_n)^T$，其 L2 范数记作 $||x||_2$，定义为 $||x||_2 = (|x_1|^2+|x_2|^2+\cdots+|x_n|^2)^{\frac{1}{2}}$ 。向量的 L2 范数即为向量中各个元素平方和的平方根，对应欧式距离 (Manhattan distance)。

**无穷范数**

假设 n 维向量 $x = (x_1,x_2,\cdots,x_n)^T$，其无穷范数记作 $||x||_\infty$，定义为 $||x||_\infty = max(|x_1|,|x_2|,\cdots,|x_n|)$ 。向量的无穷范数即为向量中各个元素绝对值的最大值，对应切比雪夫距离 (Chebyshev distance)。

#### 点到超平面距离公式的推导过程

已知 $S$ 为 n 维欧式空间中的 n-1 维超平面 $w·x + b =0$，其中 $w$ 和 $x$ 均为 n 维向量；另有 n 维空间中的点 $x_0 = (x_0^{(1)},x_0^{(2)},\cdots,x_0^{(n)})$ 。求证：点 $P$ 到超平面 $S$ 的距离 $d = \frac{1}{||w||_2} |w·x_0+b|$ ，其中 $||w||_2$ 为 $w$ 的 2-范数。

证明如下：

由超平面 $S$ 的定义式可知 $w$ 为超平面 $S$ 的法向量， $b$ 为超平面 $S$ 的截距。

设点 $x_0$ 在超平面 $S$ 上的投影为 $x_1 = (x_1^{(1)},x_1^{(2)},\cdots,x_1^{(n)})$ ，则有

$$
w · x_1 + b = 0 \tag{1}
$$

点 $P$ 到超平面 $S$ 的距离 $d$ 即为向量 $\vec{x_0 x_1}$ 的长度。

因为 $\vec{x_0 x_1}$ 与超平面 $S$ 的法向量 $w$ 平行，所以 $\vec{x_0 x_1}$ 与法向量夹角的余弦值 $cos \theta = 0$ ，故有

$$
\begin{aligned}
w · \vec{x_0 x_1}
& = |w| \ |\vec{x_0 x_1}| \ cos \theta \\
& = |w| \ |\vec{x_0 x_1}| \\
& = [(w^{(1)})^2 + (w^{(2)})^2 + \cdots + (w^{(n)})^2]^\frac{1}{2} \ d \\
& = ||w||_2 d
\end{aligned}
\tag{2}
$$

又有（应用向量点积的分配律）

$$
\begin{aligned}
w · \vec{x_0 x_1}
& = w^{(1)} (x_1^{(1)} - x_0^{(1)}) + w^{(2)} (x_1^{(2)} - x_0^{(2)}) + \cdots + w^{(n)} (x_1^{(n)} - x_0^{(n)}) \\
& = (w^{(1)} x_1^{(1)} + w^{(2)} x_1^{(2)} + \cdots + w^{(n)} x_1^{(n)}) - (w^{(1)} x_0^{(1)} + w^{(2)} x_0^{(2)} + \cdots + w^{(n)} x_0^{(n)}) \\
& = w·x_1 - w·x_0
\end{aligned}
\tag{3}
$$

由式(1)，有 $w·x_1 = -b$，故式(3)可以写成

$$
\begin{aligned}
w · \vec{x_0 x_1}
& = w·x_1 - w·x_0 \\
& = -b - w·x_0
\end{aligned}
\tag{4}
$$

由式(2)和式(4)，得

$$
\begin{aligned}
||w||_2 d = |-b - w·x_0| \\
d = \frac{1}{||w||_2} |w·x_0+b|
\end{aligned}
$$

得证。

## 2.3.1：感知机学习算法的原始形式

【补充说明】在例 2.1 的迭代过程中，选择误分类点的规则是选择索引最小的误分类点。

#### 从梯度到更新方法的说明

对于某个参数组合 $(w_0,b_0)$，因为误分类点集合 $M$ 是固定的，所以梯度 $\nabla L(w_0,b_0)$ 也是固定的，梯度在 $w$ 上的分量 $\nabla w L(w_0,b_0)$ 就是损失函数对 $w$ 的偏导数，梯度在 $b$ 上的分量 $\nabla_b L(w_0,b_0)$ 就是损失函数对 $b$ 的偏导数，于是有梯度

$$
\nabla_w L(w_0,b_0) = L'_w(w_0,b_0) = - \sum_{x_i \in M} y_i x_i
$$

$$
\nabla_b L(w_0,b_0) = L'_b(w_0,b_0) = - \sum_{x_i \in M} y_i
$$

因为完整地计算出梯度需要用到所有的样本点，时间成本较高，所以这里我们使用时间速度快的随机梯度下降法。在每次迭代过程中，不是一次使 $M$ 中所有误分类点的梯度下降，而是一次随机选取一个误分类点，使其梯度下降。对于单个误分类点 $(x_i,y_i)$，有梯度

$$
\nabla_w L(w_0,b_0) = L'_w(w_0,b_0) = - y_i x_i
$$

$$
\nabla_b L(w_0,b_0) = L'_b(w_0,b_0) = - y_i
$$

据此更新 $w$ 和 $b$：

$$
w \leftarrow w + \eta(-\nabla_w L(w_0,b_0))  = w + \eta y_i x_i
$$

$$
b \leftarrow b + \eta(-\nabla_b L(w_0,b_0)) = b + \eta y_i
$$

其中，$\eta$ 为步长，通常取值范围为 $(0,1]$，又称为学习率。

#### 感知机学习算法的原始形式（Python 实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_original_form.py)】code.perceptron.original_form_of_perceptron

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_original_form.py

def original_form_of_perceptron(x, y, eta):
    """感知机学习算法的原始形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的w和b
    """
    n_samples = len(x)  # 样本点数量
    n_features = len(x[0])  # 特征向量维度数
    w0, b0 = [0] * n_features, 0  # 选取初值w0,b0

    while True:  # 不断迭代直至没有误分类点
        for i in range(n_samples):
            xi, yi = x[i], y[i]
            if yi * (sum(w0[j] * xi[j] for j in range(n_features)) + b0) <= 0:
                w1 = [w0[j] + eta * yi * xi[j] for j in range(n_features)]
                b1 = b0 + eta * yi
                w0, b0 = w1, b1
                break
        else:
            return w0, b0
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC2%E7%AB%A0_%E6%84%9F%E7%9F%A5%E6%9C%BA/%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%9A%84%E5%8E%9F%E5%A7%8B%E5%BD%A2%E5%BC%8F.py)】测试

```python
>>> from code.perceptron import original_form_of_perceptron
>>> dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]
>>> original_form_of_perceptron(dataset[0], dataset[1], eta=1)
([1, 1], -3)
```

## 2.3.2：感知机学习算法的收敛性证明

【补充说明】扩充权重向量：扩充权重向量即将偏置并入权重向量的向量 $\hat{w} = (w^T,b)^T$。（下文直接使用了“扩充权重向量”的名称）

【补充说明】误分类次数：即存在误分类实例的迭代次数。

#### 【问题】 为什么一定存在 $||\hat{w}_{opt}||=1$？

不妨设超平面的扩充权重向量为

$$
\hat{w}'_{opt}=({w'}_{opt}^T,b')^T = (w'^{(1)}_{opt},w'^{(2)}_{opt},\cdots,w'^{(n)}_{opt},b')^T
$$

有 $||\hat{w}'_{opt}||!=1$，于是存在

$$
\hat{w}_{opt} =  (\frac{w'^{(1)}_{opt}}{||\hat{w}'_{opt}||},\frac{w'^{(2)}_{opt}}{||\hat{w}'_{opt}||},\cdots,\frac{w'^{(n)}_{opt}}{||\hat{w}'_{opt}||},\frac{b'}{||\hat{w}'_{opt}||})^T
$$

此时 $||\hat{w}_{opt}||=1$。得证。

例如：$x^{(1)}+x^{(2)}-3$ 的扩充权重向量为 $\hat{w'} = (1,1,-3)^T$，$||\hat{w'}||=\sqrt{11}$，于是有 $\hat{w}=(\frac{1}{\sqrt{11}},\frac{1}{\sqrt{11}},\frac{-3}{\sqrt{11}})^T$，使 $||\hat{w}||=1$。

## 2.3.3：感知机学习算法的对偶形式

#### 【问题】为什么要有感知机学习算法的对偶形式？

结论是感知机学习算法的对偶形式在一定条件下的运算效率更高。下面我们展开讨论。

首先考虑感知机学习算法的原始形式和对偶形式的时间复杂度。原始形式每次迭代的时间复杂度为 $O(S×N)$，总时间复杂度为 $O(S×N×K)$；对偶形式每次迭代的时间复杂度为 $O(S^2)$，另外还需要计算 Gram 矩阵的时间复杂度 $O(S^2×N)$，总时间复杂度为 $O(S^2×N+S^2×K)$；其中 $S$ 为样本点数量，$N$ 为样本特征向量的维度数，$K$ 为迭代次数。

因为原始形式和对偶形式的迭代步骤是相互对应的，所以一般来说，原始形式更适合维度少、数量高的训练数据，对偶形式更适合维度高、数量少的训练数据。

#### 计算 Gram 矩阵（Python 实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_gram.py)】code.perceptron.count_gram

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_gram.py

def count_gram(x):
    """计算Gram矩阵

    :param x: 输入变量
    :return: 输入变量的Gram矩阵
    """
    n_samples = len(x)  # 样本点数量
    n_features = len(x[0])  # 特征向量维度数
    gram = [[0] * n_samples for _ in range(n_samples)]  # 初始化Gram矩阵

    # 计算Gram矩阵
    for i in range(n_samples):
        for j in range(i, n_samples):
            gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(n_features))

    return gram
```

#### 感知机学习算法的对偶形式（Python 实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_dual_form.py)】code.perceptron.dual_form_perceptron

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/perceptron/_dual_form.py

from . import count_gram  # code.perceptron.count_gram

def dual_form_perceptron(x, y, eta):
    """感知机学习算法的对偶形式
    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的a(alpha)和b
    """
    n_samples = len(x)  # 样本点数量
    a0, b0 = [0] * n_samples, 0  # 选取初值a0(alpha),b0
    gram = count_gram(x)  # 计算Gram矩阵

    while True:  # 不断迭代直至没有误分类点
        for i in range(n_samples):
            yi = y[i]

            val = 0
            for j in range(n_samples):
                xj, yj = x[j], y[j]
                val += a0[j] * yj * gram[i][j]

            if (yi * (val + b0)) <= 0:
                a0[i] += eta
                b0 += eta * yi
                break
        else:
            return a0, b0
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC2%E7%AB%A0_%E6%84%9F%E7%9F%A5%E6%9C%BA/%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%9A%84%E5%AF%B9%E5%81%B6%E5%BD%A2%E5%BC%8F.py)】测试

```python
>>> from code.perceptron import dual_form_perceptron
>>> dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
>>> dual_form_perceptron(dataset[0], dataset[1], eta=1)
([2, 0, 5], -3)
```
