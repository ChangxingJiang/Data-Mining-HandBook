# 《统计学习方法》啃书辅助：附录A 梯度下降法

> 【补充说明】书中没有区分“梯度下降法”和“最速下降法”，但实际上两者是存在区别的。算法A.1的梯度下降法实际上是最速下降法。

#### 梯度下降法和最速下降法

在确定步长时，我们有两种思路：一种是依据瞬时变化率计算步长，在距离极值点较远时，步长取得更大一些，使算法收敛更快；而当距离极值点较近时，步长取得更小一些，避免在极小值点产生震荡；另一种是依据一维搜索计算步长，即在每次确定迭代方向后，直接移动到当前方向的**极值点**（驻点），从而避免在当前方向上的反复震荡。

第一种依据瞬时变化率计算步长的方法，就是梯度下降法；第二种依据一维搜索计算步长的方法，就是最速下降法。

#### 梯度

二元函数$z = f(x,y)$为例。如果函数在点$P_0(x_0,y_0)$可微分，$\bold{e}_l = (\cos \alpha,\cos \beta)$是与方向$l$同方向的单位向量，那么函数在点$P_0$沿方向$l$的方向导数
$$
\begin{align}
\frac{\partial f}{\partial l}|_{(x_0,y_0)} 
& = f_x(x_0,y_0) cos \ \alpha + f_y(x_0,y_0) cos \ \beta \\
& = (f_x(x_0,y_0),f_y(x_0,y_0)) · (cos \ \alpha, cos \ \beta) \\
& = (f_x(x_0,y_0),f_y(x_0,y_0)) · \bold{e}_l \\
& = |(f_x(x_0,y_0),f_y(x_0,y_0))| \ cos \ \theta \\ 
\end{align}
\tag{22}
$$
其中$\theta$是$(f_x(x_0,y_0),f_y(x_0,y_0))$和$\bold{e}_l$的交角。上式(22)表明了函数在这一点的方向导数与向量$(f_x(x_0,y_0),f_y(x_0,y_0))$之间的关系。我们将向量$(f_x(x_0,y_0),f_y(x_0,y_0))$称为函数$f(x,y)$在点$P_0(x_0,y_0)$的梯度，记作$\bold{grad} \ f(x_0,y_0)$或$\nabla f(x_0,y_0)$。特别的有：

* 当方向$\bold{e}_l$与梯度$\bold{grad} \ f(x_0,y_0)$方向一致时，函数$f(x,y)$增加最快，即函数在这个方向的方向导数取得最大值，这个最大值为梯度$\bold{grad} \ f(x_0,y_0)$的模；
* 当方向$\bold{e}_l$与梯度$\bold{grad} \ f(x_0,y_0)$方向相反时，函数$f(x,y)$减少最快，即函数在这个方向的方向导数取得最小值，这个最小值为梯度$\bold{grad} \ f(x_0,y_0)$的模的相反数；
* 当方向$\bold{e}_l$与梯度$\bold{grad} \ f(x_0,y_0)$方向正交时，函数$f(x,y)$的变化率为零。

在同济大学的《高等数学》中，对二元函数的**梯度**，给出如下定义：

> ##### 【定义】梯度（来自同济大学《高等数学》第七版下册 P. 106）
>
> 设函数$f(x,y)$在平面区域$D$内具有一阶连续偏导数，则对于每一点$P_0(x_0,y_0) \in D$，都可定出一个向量
>
> $$
> f_x(x_0,y_0) \bold{i} + f_y(x_0,y_0) \bold{j} \tag{22}
> $$
>
> 这向量称为函数$f(x,y)$在点$P_0(x_0,y_0)$的梯度，记作$\bold{grad} \ f(x_0,y_0)$或$\nabla f(x_0,y_0)$，即
> $$
> \bold{grad} \ f(x_0,y_0) = \nabla f(x_0,y_0) = f_x(x_0,y_0) \bold{i} + f_y(x_0,y_0) \bold{j} \tag{23}
> $$
>
> 其中$\nabla = \frac{\partial}{\partial x} \bold{i} + \frac{\partial}{\partial y} \bold{j}$称为（二维的）向量微分算子或Nabla算子，$\nabla f = \frac{\partial f}{\partial x} \bold{i} + \frac{\partial f}{\partial y} \bold{j}$。

其中$\bold{i}$、$\bold{j}$分别为与第一坐标轴同方向和与第二坐标轴同方向的单位向量。

#### 梯度向量计算（Python+scipy计算）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_partial_derivative.py)】code.gradient_descent.partial_derivative

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_partial_derivative.py

from scipy.misc import derivative

def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点各个自变量的梯度向量（偏导数列表）

    :param func: [function] n元函数
    :param arr: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 偏导数
    """
    n_features = len(arr)
    ans = []
    for i in range(n_features):
        def f(x):
            arr2 = list(arr)
            arr2[i] = x
            return func(arr2)

        ans.append(derivative(f, arr[i], dx=dx))
    return ans
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95A_%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95/%E6%A2%AF%E5%BA%A6%E5%90%91%E9%87%8F%E8%AE%A1%E7%AE%97.py)】测试

```python
>>> from code.gradient_descent import partial_derivative
>>> partial_derivative(lambda x: x[0] ** 2, [3])
[6.000000000838668]
>>> partial_derivative(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, [0, 0])
[3.000000000419334, 3.9999999996709334]
```

#### 基于黄金分割法的一维搜索（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_golden_section_for_line_search.py)】code.gradient_descent.golden_section_for_line_search

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_golden_section_for_line_search.py

def golden_section_for_line_search(func, a0, b0, epsilon):
    """一维搜索极小值点（黄金分割法）

    :param func: [function] 一元函数
    :param a0: [int/float] 目标区域左侧边界
    :param b0: [int/float] 目标区域右侧边界
    :param epsilon: [int/float] 精度
    """
    a1, b1 = a0 + 0.382 * (b0 - a0), b0 - 0.382 * (b0 - a0)
    fa, fb = func(a1), func(b1)

    while b1 - a1 > epsilon:
        if fa <= fb:
            b0, b1, fb = b1, a1, fa
            a1 = a0 + 0.382 * (b0 - a0)
            fa = func(a1)
        else:
            a0, a1, fa = a1, b1, fb
            b1 = b0 - 0.382 * (b0 - a0)
            fb = func(b1)

    return (a1 + b1) / 2
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95A_%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95/%E5%9F%BA%E4%BA%8E%E9%BB%84%E9%87%91%E5%88%86%E5%89%B2%E6%B3%95%E7%9A%84%E4%B8%80%E7%BB%B4%E6%90%9C%E7%B4%A2.py)】测试

```python
>>> from code.gradient_descent import golden_section_for_line_search
>>> golden_section_for_line_search(lambda x: x ** 2, -10, 5, epsilon=1e-6)
5.263005013597177e-06
```

#### 梯度下降法（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_gradient_descent.py)】code.gradient_descent.gradient_descent

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_gradient_descent.py

from ._partial_derivative import partial_derivative  # code.gradient_descent.partial_derivative

def gradient_descent(func, n_features, eta, epsilon, maximum=1000):
    """梯度下降法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param eta: [int/float] 学习率
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取自变量初值
    y0 = func(x0)  # 计算函数值
    for _ in range(maximum):
        nabla = partial_derivative(func, x0)  # 计算梯度
        x1 = [x0[i] - eta * nabla[i] for i in range(n_features)]  # 迭代自变量
        y1 = func(x1)  # 计算函数值
        if abs(y1 - y0) < epsilon:  # 如果当前变化量小于学习精度，则结束学习
            return x1
        x0, y0 = x1, y1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95A_%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.py)】测试

```python
>>> from code.gradient_descent import gradient_descent
>>> gradient_descent(lambda x: x[0] ** 2, 1, eta=0.1, epsilon=1e-6)
[0.0]
>>> gradient_descent(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, eta=0.1, epsilon=1e-6)
[-2.9983082373813077, -3.997744316508843]
```

#### 最速下降法（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_steepest_descent.py)】code.gradient_descent.steepest_descent

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/gradient_descent/_steepest_descent.py

from ._golden_section_for_line_search import golden_section_for_line_search  # code.gradient_descent.golden_section_for_line_search
from ._partial_derivative import partial_derivative  # code.gradient_descent.partial_derivative

def steepest_descent(func, n_features, epsilon, distance=3, maximum=1000):
    """梯度下降法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取自变量初值
    y0 = func(x0)  # 计算函数值
    for _ in range(maximum):
        nabla = partial_derivative(func, x0)  # 计算梯度

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        def f(x):
            """梯度方向的一维函数"""
            x2 = [x0[i] - x * nabla[i] for i in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # 一维搜索寻找驻点

        x1 = [x0[i] - lk * nabla[i] for i in range(n_features)]  # 迭代自变量
        y1 = func(x1)  # 计算函数值

        if abs(y1 - y0) < epsilon:  # 如果当前变化量小于学习精度，则结束学习
            return x1

        x0, y0 = x1, y1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95A_%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95/%E6%9C%80%E9%80%9F%E4%B8%8B%E9%99%8D%E6%B3%95.py)】测试

```python
>>> from code.gradient_descent import steepest_descent
>>> steepest_descent(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>> steepest_descent(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-2.9999999999635865, -3.999999999951452]
```

