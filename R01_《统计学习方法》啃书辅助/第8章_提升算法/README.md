# 《统计学习方法》啃书辅助：第8章 提升算法

## 8.1.1 提升方法的基本思路

#### PAC 学习框架

假设输入空间是 $\mathcal{X} \in R^n$，输入 $x \in \mathcal{X}$ 表示实例的特征向量。例如，每个实例都是一个人，$\mathcal{X}$ 表示所有实例点的集合。

对于二类分类问题，有输出变量 $\mathcal{Y} = \{-1,+1\}$。概念 $c:\mathcal{X} \rightarrow \mathcal{Y}$ 是指一个从 $\mathcal{X}$ 到 $\mathcal{Y}$ 的映射，即布尔函数 $\mathcal{X} \rightarrow \{-1,+1\}$；也可以用输入空间 $\mathcal{X}$ 中对应输出变量值为 $1$ 的子集来定义这个概念。例如，这个概念可以是“吃过螃蟹的人”。

概念类是指我们希望学习的所有可能概念的集合，记作 $C$。

## 8.1.2 AdaBoost 算法

【补充说明】步骤(2).(c)：当 $e_m \rightarrow 0$ 时，$\alpha_m \rightarrow +\infty$；当 $e_m = \frac{1}{2}$ 时，$\alpha_m = 0$；当 $e_m \rightarrow 1$ 时，$\alpha_m \rightarrow -\infty$。由此可见，当 $e_m \le \frac{1}{2}$ 时，分类误差率越小的基本分类器在最终分类器中的作用越大。

#### AdaBoost 算法（Python 实现）

```python
import math
from copy import copy
import numpy as np

class AdaBoost:
    """AdaBoost算法

    :param X: 输入变量
    :param Y: 输出变量
    :param weak_clf: 弱分类算法
    :param M: 弱分类器数量
    """

    def __init__(self, X, Y, weak_clf, M=10):
        self.X, self.Y = X, Y
        self.weak_clf = weak_clf
        self.M = M

        # ---------- 初始化计算 ----------
        self.n_samples = len(self.X)
        self.n_features = len(self.X[0])

        # ---------- 取初值 ----------
        self.G_list = []  # 基本分类器列表
        self.a_list = []  # 分类器系数列表

        # ---------- 执行训练 ----------
        self._train()

    def _train(self):
        # 初始化训练数据的权值分布
        D = [1 / self.n_samples] * self.n_samples

        # 当前所有弱分类器的线性组合的预测结果
        fx = [0] * self.n_samples

        # 迭代增加弱分类器
        for m in range(self.M):
            # 使用具有权值分布D的训练数据集学习，得到基本分类器
            self.weak_clf.fit(self.X, self.Y, sample_weight=D)

            # 使用Gm(x)预测训练数据集的所有实例点
            predict = self.weak_clf.predict(self.X)

            # 计算Gm(x)在训练数据集上的分类误差率
            error = sum(D[i] for i in range(self.n_samples) if np.sign(predict[i]) != self.Y[i])

            # 计算Gm(x)的系数
            a = 0.5 * math.log((1 - error) / error, math.e)

            self.G_list.append(copy(self.weak_clf))
            self.a_list.append(a)

            # 更新训练数据集的权值分布
            D = [D[i] * pow(math.e, -a * self.Y[i] * predict[i]) for i in range(self.n_samples)]
            Z = sum(D)  # 计算规范化因子
            D = [v / Z for v in D]

            # 计算当前所有弱分类器的线性组合的误分类点数
            wrong_num = 0
            for i in range(self.n_samples):
                fx[i] += a * predict[i]  # 累加当前所有弱分类器的线性组合的预测结果
                if np.sign(fx[i]) != self.Y[i]:
                    wrong_num += 1

            print("迭代次数:", m + 1, ";", "误分类点数:", wrong_num)

            # 如果当前所有弱分类器的线性组合已经没有误分类点，则结束迭代
            if wrong_num == 0:
                break

    def predict(self, x):
        """预测实例"""
        return np.sign(sum(self.a_list[i] * self.G_list[i].predict([x]) for i in range(len(self.G_list))))
```

【源码地址】例 8.1 测试

```python
from sklearn.tree import DecisionTreeClassifier
from code.adaboost import AdaBoost

if __name__ == "__main__":
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]]

    clf = AdaBoost(dataset[0], dataset[1], DecisionTreeClassifier(max_depth=1))
    correct = 0
    for ii in range(10):
        if clf.predict(dataset[0][ii]) == dataset[1][ii]:
            correct += 1
    print("预测正确率:", correct / 10)  # 预测正确率: 1.0 (3次迭代)
```

【源码地址】乳腺癌数据集测试

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from code.adaboost import AdaBoost

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoost(x1, y1, DecisionTreeClassifier(max_depth=1))
    correct = 0
    for i in range(len(x2)):
        if clf.predict(x2[i]) == y2[i]:
            correct += 1
    print("预测正确率:", correct / len(x2))  # 预测正确率: 0.9157894736842105 (10+次迭代)
```

#### AdaBoost 算法（sklearn 实现）

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
    clf.fit(x1, y1)
    print("预测正确率:", clf.score(x2, y2))  # 预测正确率: 0.9736842105263158
```

## 8.2 AdaBoost 算法的训练误差分析

【补充说明】在定理 8.1 的推导过程中：从第一行到第二行需要用到 $w_{1i}=\frac{1}{N}$。第二行及此后的推导需要用到 $Z_m$ 的变形式。

【扩展阅读】[统计学习方法（7）前向分步算法推导 AdaBoost 的详细过程](https://blog.csdn.net/olizxq/article/details/89400466)

#### 定理 8.2 的推导（完整详细过程）

由 $Z_m$ 的定义式（书中式 8.5）有：

$$
Z_m = \sum_{i=1}^N w_{mi} \ exp(-\alpha_m y_i G_m(x_i)) \tag{1}
$$

考虑到，当 $G_m(x)$ 分类正确时，有 $y_i g_m(x_i) = 1$；分类错误时，有 $y_i g_m(x_i) = -1$。于是上式(1)可以写成：

$$
Z_m = \sum_{y_i = G_m(x_i)} w_{mi} exp(-\alpha_m) + \sum_{y_i \ne G_m(x_i)} w_{mi} exp(\alpha_m) \tag{2}
$$

根据基本分类器 $G_m(x)$ 的分类误差率（书中式 8.8），上式(2)可以写成：

$$
Z_m = (1-e_m) exp(-\alpha_m) + e_m exp(\alpha_m) \tag{3}
$$

将 $\alpha_m$ 的定义式（书中式 8.2）代入上式(3)，则有:

$$
\begin{aligned}
Z_m
& = (1-e_m) exp(- \frac{1}{2} log \frac{1-e_m}{e_m}) + e_m exp(\frac{1}{2} log \frac{1-e_m}{e_m}) \\
& = (1-e_m) exp(\frac{1}{2} log \frac{e_m}{1-e_m}) + e_m exp(\frac{1}{2} log \frac{1-e_m}{e_m}) \\
& = (1-e_m) \sqrt{\frac{e_m}{1-e_m}} + e_m \sqrt{\frac{1-e_m}{e_m}} \\
& = 2 \sqrt{e_m(1-e_m)}
\end{aligned} \tag{4}
$$

令 $\gamma_m = \frac{1}{2} - e_m$，上式(4)可以写成：

$$
Z_m = \sqrt{1 - 4 \gamma_m^2} \tag{5}
$$

设 $f(x) = \sqrt{1-x}$，则有 $f(x)$ 在点 $x=0$ 的二阶泰勒展开式

$$
f(x) \approx f(0) + x f'(0) + \frac{1}{2} x^2 f''(0) = 1 - \frac{1}{2} x - \frac{1}{8} x^2
$$

令 $x=4\gamma_m^2$，有

$$
\sqrt{1-4\gamma_m^2} \approx f(4\gamma_m^2) = 1-2\gamma_m^2 - 2\gamma_m^4 \tag{6}
$$

设 $g(x) = e^2$，则有 $g(x)$ 在点 $x=0$ 的二阶泰勒展开式

$$
g(x) \approx g(0) + x g'(0) + \frac{1}{2} x^2 g''(0) = 1 + x + \frac{1}{2} x^2
$$

令 $x=-2\gamma_m^2$，有

$$
exp(-2\gamma_m^2) \approx g(-2\gamma_m^2) = 1 -2\gamma_m^2 + 2\gamma_m^4 \tag{7}
$$

根据式(6)和式(7)，有

$$
\sqrt{1-4\gamma_m^2} \approx 1-2\gamma_m^2 - 2\gamma_m^4 \le 1 -2\gamma_m^2 + 2\gamma_m^4 \approx exp(-2\gamma_m^2) \tag{8}
$$

于是有

$$
\prod_{m=1}^M \sqrt{1-4\gamma_m^2} \le \prod_{m=1}^M exp(-2\gamma_m^2) = exp(-2 \sum_{m=1}^M \gamma_m^2) \tag{9}
$$

得证。

## 8.3 AdaBoost 算法的解释

#### 定理 8.3 的推导（求 $\alpha_m^*$ 部分详细过程）

将式（8.21）中以 $\alpha$ 和 $G(x)$ 为变量的指数损失函数设为 $L(\alpha,G(x))$，参考定理 8.2 的推导过程有

$$
\begin{aligned}
L(\alpha,G(x))
& = \sum_{i=1}^N \bar{w}_{mi} exp [-y_i \alpha G(x_i)] \\
& = \sum_{y_i = G_m(x_i)} \bar{w}_{mi} e^{-\alpha} + \sum_{y_i \ne G_m(x_i)} \bar{w}_{mi} e^\alpha \\
& = \sum_{i=1}^N \bar{w}_{mi} e^{-\alpha} - \sum_{y_i \ne G_m(x_i)} \bar{w}_{mi} e^{-\alpha} + \sum_{y_i \ne G_m(x_i)} \bar{w}_{mi} e^\alpha \\
& = (e^\alpha - e^{-\alpha}) \sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) + e^{-\alpha} \sum_{i=1}^N \bar{w}_{mi}
\end{aligned} \tag{1}
$$

将已求得的 $G_m^*(x)$ 代入上式，即在 $G_m(x)$ 已知的情况下，对 $\alpha$ 求导并令导数为 0，得到

$$
\frac{\partial L}{\partial \alpha} = e^\alpha \sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) + e^{-\alpha} \sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) - e^{-\alpha} \sum_{i=1}^N \bar{w}_{mi} = 0 \tag{2}
$$

对上式(2)整理得到

$$
e^\alpha \sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) = e^{-\alpha} \Big{[} \sum_{i=1}^N \bar{w}_{mi} -\sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) \Big{]} \tag{3}
$$

对上式(3)两边同时求对数，得到

$$
log \ e^\alpha + log \sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) = log \ e^{-\alpha} + log \Big{[} \sum_{i=1}^N \bar{w}_{mi} -\sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i)) \Big{]} \tag{4}
$$

因为 $log \ e^{-\alpha} = - log \ e^\alpha$，于是上式(4)可以写成

$$
log \ e^\alpha = \frac{1}{2} log \frac{\sum_{i=1}^N \bar{w}_{mi} -\sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i))}{\sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i))} \tag{5}
$$

根据 $e_m = \frac{\sum_{i=1}^N \bar{w}_{mi} I(y_i \ne G(x_i))}{\sum_{i=1}^N \bar{w}_{mi}}$，令上式(5)中的分子分母同除 $\sum_{i=1}^N \bar{w}_{mi}$，得到

$$
\alpha_m = \frac{1}{2} log \frac{1-e_m}{e_m}
$$

## 8.4 提升树

【重点】提升树模型可以表示为决策树的加法模型。

【补充说明】梯度提升算法的学习率，是指将新拟合的模型更新到模型时的权重，在书中默认为 1，但这个值实际是可以设置的。

#### 回归问题的提升树算法（原生 Python 实现）

```python
from copy import copy

from sklearn.tree import DecisionTreeRegressor


class AdaBoostRegressor:
    """AdaBoost算法

    :param X: 输入变量
    :param Y: 输出变量
    :param weak_reg: 基函数
    :param M: 基函数的数量
    """

    def __init__(self, X, Y, weak_reg, M=10):
        self.X, self.Y = X, Y
        self.weak_reg = weak_reg
        self.M = M

        self.n_samples = len(self.X)
        self.G_list = []  # 基函数列表

        # ---------- 执行训练 ----------
        self._train()

    def _train(self):
        # 计算当前的残差：f(x)=0时
        r = [self.Y[i] for i in range(self.n_samples)]

        # 迭代增加基函数
        for m in range(self.M):
            # 拟合残差学习一个基函数
            self.weak_reg.fit(self.X, r)

            self.G_list.append(copy(self.weak_reg))

            # 计算更新后的新残差
            predict = self.weak_reg.predict(self.X)
            for i in range(self.n_samples):
                r[i] -= predict[i]

    def predict(self, x):
        """预测实例"""
        return sum(self.G_list[i].predict([x])[0] for i in range(len(self.G_list)))
```

【源码地址】例 8.2 测试

```python
from sklearn.tree import DecisionTreeRegressor
from code.adaboost import AdaBoostRegressor

if __name__ == "__main__":
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]]

    seg = AdaBoostRegressor(dataset[0], dataset[1], DecisionTreeRegressor(max_depth=1), M=6)
    r = sum((seg.predict(dataset[0][i]) - dataset[1][i]) ** 2 for i in range(10))
    print("平方误差损失:", r)  # 平方误差损失: 0.17217806498628369
```

【源码地址】波士顿房价数据集

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from code.adaboost import AdaBoostRegressor

if __name__ == "__main__":
    X, Y = load_boston(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = AdaBoostRegressor(x1, y1, DecisionTreeRegressor(max_depth=1), M=50)
    r = sum((seg.predict(x2[i]) - y2[i]) ** 2 for i in range(len(x2)))
    print("平方误差损失:", r)  # 平方误差损失: 3880.770754880455
```

#### 回归问题的提升树算法（sklearn 实现）

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, Y = load_boston(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = GradientBoostingRegressor(n_estimators=50, learning_rate=1, max_depth=1, random_state=0, loss='ls')
    seg.fit(x1, y1)
    r = sum((seg.predict([x2[i]])[0] - y2[i]) ** 2 for i in range(len(x2)))
    print("平方误差损失:", r)  # 平方误差损失: 3880.770754880452
```

#### 梯度下降法和梯度提升算法的区别

梯度下降法和梯度提升算法都是在求解损失函数最小化的问题。在梯度下降法中，我们求解的变量是损失函数参数 $\theta$，而在梯度提升算法中，我们求解的变量是函数 $F(x)$。

在梯度下降法的每次迭代中，首先计算出损失函数的负梯度在当前参数的值，然后向着该值的方向更新参数（固定步长或一维搜索）。

在梯度提升算法的每次迭代中，首先计算出损失函数的负梯度在当前模型的值，然后以该值为目标，拟合一个基函数，从而实现对模型的更新。
