# 《统计学习方法》啃书辅助：第2章 感知机

**感知机模型对数据的要求**：训练数据集中需存在某个超平面能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即训练数据集是线性可分的。因为只有当训练数据集是线性可分时，感知机学习算法才是收敛的；如果训练数据集线性不可分，则感知机学习算法不收敛，迭代结果会发生震荡。当训练数据集线性不可分时，可以使用线性支持向量机。

**感知机模型的学习过程**：依据训练数据集求得感知机模型，即求得模型参数$w$和$b$。

**感知机模型的预测过程**：通过学习得到的感知机模型，计算新的输入实例所对应的输出类别。

**感知机模型的类别划分**：

* 用于解决二类分类问题的监督学习模型
* 非概率模型：模型取函数形式（而非概率模型的条件概率分布形式）
* 线性模型：模型函数为线性函数
* 参数化模型：模型参数的维度固定
* 判别模型：直接学习决策函数$f(x)$

**感知机模型的主要优点**：算法简单，易于实现。

**感知机模型的主要缺点**：要求训练数据集线性可分。

-----

## 2.2 感知机学习策略

【延伸知识】[常见范数的定义和性质](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC2%E7%AB%A0_%E6%84%9F%E7%9F%A5%E6%9C%BA/%E3%80%90%E9%85%8D%E5%A5%97%E7%9F%A5%E8%AF%86%E3%80%91%E5%B8%B8%E8%A7%81%E8%8C%83%E6%95%B0%E7%9A%84%E5%AE%9A%E4%B9%89%E5%92%8C%E6%80%A7%E8%B4%A8.md)

【延伸知识】[点到超平面距离公式的推导过程](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC2%E7%AB%A0_%E6%84%9F%E7%9F%A5%E6%9C%BA/%E3%80%90%E9%85%8D%E5%A5%97%E7%9F%A5%E8%AF%86%E3%80%91%E7%82%B9%E5%88%B0%E8%B6%85%E5%B9%B3%E9%9D%A2%E8%B7%9D%E7%A6%BB%E5%85%AC%E5%BC%8F%E7%9A%84%E6%8E%A8%E5%AF%BC%E8%BF%87%E7%A8%8B.md)

## 2.3.1：感知机学习算法的原始形式

> 【补充说明】在例2.1的迭代过程中，选择误分类点的规则是选择索引最小的误分类点。

#### 从梯度到更新方法的说明

对于某个参数组合$(w_0,b_0)$，因为误分类点集合$M$是固定的，所以梯度$\nabla L(w_0,b_0)$也是固定的，梯度在$w$上的分量$\nabla w L(w_0,b_0)$就是损失函数对$w$的偏导数，梯度在$b$上的分量$\nabla_b L(w_0,b_0)$就是损失函数对$b$的偏导数，于是有梯度

$$
\nabla_w L(w_0,b_0) = L'_w(w_0,b_0) = - \sum_{x_i \in M} y_i x_i
$$

$$
\nabla_b L(w_0,b_0) = L'_b(w_0,b_0) = - \sum_{x_i \in M} y_i
$$

因为完整地计算出梯度需要用到所有的样本点，时间成本较高，所以这里我们使用时间速度快的随机梯度下降法。在每次迭代过程中，不是一次使$M$中所有误分类点的梯度下降，而是一次随机选取一个误分类点，使其梯度下降。对于单个误分类点$(x_i,y_i)$，有梯度

$$
\nabla_w L(w_0,b_0) = L'_w(w_0,b_0) = - y_i x_i
$$

$$
\nabla_b L(w_0,b_0) = L'_b(w_0,b_0) = - y_i
$$

据此更新$w$和$b$：

$$
w \leftarrow w + \eta(-\nabla_w L(w_0,b_0))  = w + \eta y_i x_i
$$

$$
b \leftarrow b + \eta(-\nabla_b L(w_0,b_0)) = b + \eta y_i
$$

其中，$\eta$为步长，通常取值范围为$(0,1]$，又称为学习率。

#### 感知机学习算法的原始形式（Python实现）

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

> 【补充说明】扩充权重向量：扩充权重向量即将偏置并入权重向量的向量$\hat{w} = (w^T,b)^T$。（下文直接使用了“扩充权重向量”的名称）>

> 【补充说明】误分类次数：即存在误分类实例的迭代次数。

#### 【问题】 为什么一定存在$||\hat{w}_{opt}||=1$？

不妨设超平面的扩充权重向量为
$$
\hat{w}'_{opt}=({w'}_{opt}^T,b')^T = (w'^{(1)}_{opt},w'^{(2)}_{opt},\cdots,w'^{(n)}_{opt},b')^T
$$
有$||\hat{w}'_{opt}||!=1$，于是有
$$
\hat{w}_{opt} =  (\frac{w'^{(1)}_{opt}}{||\hat{w}'_{opt}||},\frac{w'^{(2)}_{opt}}{||\hat{w}'_{opt}||},\cdots,\frac{w'^{(n)}_{opt}}{||\hat{w}'_{opt}||},\frac{b'}{||\hat{w}'_{opt}||})^T
$$
此时，存在$\hat{w}_{opt}$令$||\hat{w}_{opt}||=1$。得证。

例如：$x^{(1)}+x^{(2)}-3$的扩充权重向量为$\hat{w'} = (1,1,-3)^T$，$||\hat{w'}||=\sqrt{11}$，于是有$\hat{w}=(\frac{1}{\sqrt{11}},\frac{1}{\sqrt{11}},\frac{-3}{\sqrt{11}})^T$，使$||\hat{w}||=1$。

## 2.3.3：感知机学习算法的对偶形式

#### **【问题】**为什么要有感知机学习算法的对偶形式？

结论是感知机学习算法的对偶形式在一定条件下的运算效率更高。下面我们展开讨论。

首先考虑感知机学习算法的原始形式和对偶形式的时间复杂度。原始形式每次迭代的时间复杂度为$O(S×N)$，总时间复杂度为$O(S×N×K)$；对偶形式每次迭代的时间复杂度为$O(S^2)$，另外还需要计算Gram矩阵的时间复杂度$O(S^2×N)$，总时间复杂度为$O(S^2×N+S^2×K)$；其中$S$为样本点数量，$N$为样本特征向量的维度数，$K$为迭代次数。

因为原始形式和对偶形式的迭代步骤是相互对应的，所以一般来说，原始形式更适合维度少、数量高的训练数据，对偶形式更适合维度高、数量少的训练数据。

#### 计算Gram矩阵（Python实现）

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

#### 感知机学习算法的对偶形式（Python实现）

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

