# 《统计学习方法》啃书辅助：附录B 牛顿法和拟牛顿法

## B.1 牛顿法

【基础知识】梯度

【基础知识】[浅谈「正定矩阵」和「半正定矩阵」 - Xinyu Chen的文章 - 知乎](https://zhuanlan.zhihu.com/p/44860862)

#### 二阶泰勒展开和黑塞矩阵

若一元函数$f(x)$具有二阶连续导数，则$f(x)$在点$x_0$处的二阶泰勒展开式如下（省略高阶无穷小）：
$$
f(x) = f(x_0) + f'(x_0) \Delta x + \frac{1}{2} f''(x_0) (\Delta x)^2
$$
其中$\Delta x=x-x_0$。

若二元函数$f(x^{(1)},x^{(2)})$具有二阶连续偏导数，则$f(x^{(1)},x^{(2)})$在点$(x_0^{(1)},x_0^{(2)})$处的二阶泰勒展开式如下（省略高阶无穷小）：
$$
\begin{align}
f(x^{(1)},x^{(2)}) = &
f(x_0^{(1)},x_0^{(2)}) +
\Bigg{(}
\frac{\partial f}{\partial x^{(1)}} \bigg{|}_{(x_0^{(1)},x_0^{(2)})} \Delta x^{(1)}
+
\frac{\partial f}{\partial x^{(2)}} \bigg{|}_{(x_0^{(1)},x_0^{(2)})} \Delta x^{(2)}
\Bigg{)}
+
\\ &
\frac{1}{2}
\Bigg{(}
\frac{\partial^2 f}{\partial^2 x^{(1)}} \bigg{|}_{(x_0^{(1)},x_0^{(2)})} \big{(} \Delta x^{(1)} \big{)}^2
+
2 \frac{\partial^2 f}{\partial x^{(1)} \partial x^{(2)}} \bigg{|}_{(x_0^{(1)},x_0^{(2)})} \Delta x^{(1)} \Delta x^{(2)}
+
\frac{\partial^2 f}{\partial^2 x^{(2)}} \bigg{|}_{(x_0^{(1)},x_0^{(2)})} \big{(} \Delta x^{(2)} \big{)}^2
\Bigg{)}
+
o
\end{align}
$$
其中$\Delta x^{(1)} = x^{(1)} - x_0^{(1)}$，$\Delta x^{(2)} = x^{(2)} - x_0^{(2)}$。将上述展开式写成矩阵形式，有
$$
f(X) = f(X_0)
+ 
\begin{bmatrix}
\frac{\partial f}{\partial x^{(1)}} & \frac{\partial f}{\partial x^{(2)}}
\end{bmatrix}
\bigg{|}_{X_0}
\begin{bmatrix}
\Delta x^{(1)} \\
\Delta x^{(2)}
\end{bmatrix}
+
\frac{1}{2}
\begin{bmatrix}
\Delta x^{(1)} &
\Delta x^{(2)}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial^2 f}{\partial^2 x^{(1)}} & \frac{\partial^2 f}{\partial x^{(1)} \partial x^{(2)}} \\
\frac{\partial^2 f}{\partial x^{(2)} \partial x^{(1)}} & \frac{\partial^2 f}{\partial^2 x^{(2)}}
\end{bmatrix}
\Bigg{|}_{X_0}
\begin{bmatrix}
\Delta x^{(1)} \\
\Delta x^{(2)}
\end{bmatrix}
$$
其中$\begin{bmatrix}\frac{\partial f}{\partial x^{(1)}} & \frac{\partial f}{\partial x^{(2)}}\end{bmatrix}$是$f(X)$在点$X_0$的梯度向量的转置；$\begin{bmatrix}\frac{\partial^2 f}{\partial^2 x^{(1)}} & \frac{\partial^2 f}{\partial x^{(1)} \partial x^{(2)}} \\ \frac{\partial^2 f}{\partial x^{(2)} \partial x^{(1)}} & \frac{\partial^2 f}{\partial^2 x^{(2)}}\end{bmatrix}$是$f(x)$的黑塞矩阵在点$X_0$的值，记作$H(X_0)$。于是上式可以写成：
$$
f(X) = f(X_0) + \nabla f(X_0) ^T \Delta X + \frac{1}{2} \Delta X^T H(X_0) \Delta X
$$
以上结果可以推广到三元及三元以上的多元函数，多元函数的黑塞矩阵为：
$$
H(f) = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial^2 x^{(1)}} & \frac{\partial^2 f}{\partial x^{(1)} \partial x^{(2)}} & \cdots & \frac{\partial^2 f}{\partial x^{(1)} \partial x^{(n)}} \\
\frac{\partial^2 f}{\partial x^{(2)} \partial^{(1)}} & \frac{\partial^2 f}{\partial^2 x^{(2)}} & \cdots & \frac{\partial^2 f}{\partial x^{(2)} \partial x^{(n)}} \\
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial^2 f}{\partial x^{(n)} \partial x^{(1)}} & \frac{\partial^2 f}{\partial x^{(n)} \partial x^{(2)}} & \cdots & \frac{\partial^2 f}{\partial^2 x^{(n)}}
\end{bmatrix}
$$

#### 黑塞矩阵的计算（Python+scipy实现）

> **【定理】 （来自同济大学《高等数学》第七版下册 P. 70）**
>
> 如果函数$z=f(x,y)$的两个二阶混合偏导数$\frac{\partial^2 z}{\partial x \partial y}$及$\frac{\partial^2 z}{\partial y \partial x}$在区域D内连续，那么在该区域内这两个二阶混合偏导数必相等。

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_get_hessian.py)】code.newton_method.get_hessian

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_get_hessian.py

from scipy.misc import derivative

def get_hessian(func, x0, dx=1e-6):
    """计算n元函数在某点的黑塞矩阵

    :param func: [function] n元函数
    :param x0: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 黑塞矩阵
    """
    n_features = len(x0)
    ans = [[0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        def f1(xi, x1):
            x2 = list(x1)
            x2[i] = xi
            return func(x2)

        for j in range(n_features):
            # 当x[j]=xj时，x[i]方向的斜率
            def f2(xj):
                x1 = list(x0)
                x1[j] = xj
                res = derivative(f1, x0=x1[i], dx=dx, args=(x1,))
                return res

            ans[i][j] = derivative(f2, x0[j], dx=dx)

    return ans
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/%E8%AE%A1%E7%AE%97%E9%BB%91%E5%A1%9E%E7%9F%A9%E9%98%B5.py)】测试

```python
>>> from code.newton_method import get_hessian
>>> get_hessian(lambda x: (x[0] ** 3) * (x[1] ** 2) - 3 * x[0] * (x[1] ** 3) - x[0] * x[1] + 1, [0, 2])
[[0 ,  -37], [-37, 0]]
>>> get_hessian(lambda x: (x[0] ** 3) * (x[1] ** 2) - 3 * x[0] * (x[1] ** 3) - x[0] * x[1] + 1, [1, 1])
[[6 ,  -4], [-4, -16]]
```

#### 计算逆矩阵（Python+numpy实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/%E8%AE%A1%E7%AE%97%E9%80%86%E7%9F%A9%E9%98%B5.py)】测试

```python
>>> import numpy as np
>>> mat = np.matrix([[6, -4], [-4, -16]])
>>> mat ** -1
[[ 0.14285714 -0.03571429]
 [-0.03571429 -0.05357143]]
```

#### 牛顿法（Python+numpy+scipy实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_newton_method.py)】code.newton_method.newton_method

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_newton_method.py

import numpy as np
from ._get_hessian import get_hessian  # code.newton_method.get_hessian
from ..gradient_descent import partial_derivative  # code.gradient_descent.partial_derivative

def newton_method(func, n_features, epsilon=1e-6, maximum=1000):
    """牛顿法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        gk = np.matrix([nabla])

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算黑塞矩阵
        hessian = np.matrix(get_hessian(func, x0))

        # 计算步长 pk
        pk = - (hessian ** -1) * gk.T

        # 迭代
        for j in range(n_features):
            x0[j] += float(pk[j][0])
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/%E7%89%9B%E9%A1%BF%E6%B3%95.py)】测试

```python
>>> from code.newton_method import newton_method
>>> newton_method(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>>> newton_method(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-2.998150057576512, -3.997780069092481]
```

## B.2 拟牛顿法的思路

> 【补充说明】B14推导到B.15时，**近似**是因为忽略了二阶泰勒展开式。

#### 【问题】B.16的拟牛顿条件中为什么可以是$G_{k+1}$，而不是$G_k$？

设经过$k+1$次迭代后得到$x^{(k+1)}$，此时将目标函数$f(x)$在$x^{(k+1)}$处做二阶泰勒展开，得（类似于B.2）：
$$
f(x) =  f(x^{(k+1)}) + g_{(k+1)}^T (x-x^{(k+1)}) + \frac{1}{2} (x-x^{(k+1)})^T H(x^{(k+1)}) (x-x^{(k+1)})
$$
根据二阶导数的定义，于是有（类似于B.6）
$$
\nabla f(x) = g_{k+1} + H_{k+1} (x-x^{(k+1)})
$$
在上式中取$x = x^{(k)}$（$x^{(k)}$也在$x^{(k+1)}$的邻域内），即得（类似于B.11）
$$
g_k - g_{k+1} = H_{k+1} (x_k-x^{(k+1)})
$$
记$y_k=g_{k+1} - g_k$，$\delta_k = x^{(k+1)}-x^{(k)}$，则（类似于B.13）
$$
H_{k+1}^{-1} y_k = \delta_k
$$

## B.3  DFP算法

> 【补充说明】正定对称的初始矩阵$G_0$，不妨取单位矩阵$I$。

#### 【问题】DFP算法和BFGS算法等为什么需要一维搜索？

当目标函数是二次函数时，因为二阶泰勒展开函数与原目标函数是完全相同的，所以从任一初始点出发，只需要一次迭代即可达到极小值点。因此，牛顿法是一种具有二次收敛性的算法。对于非二次函数，若函数的二次性态较强，或迭代点已进入极小点的邻域，则其收敛速度是很快的。

但是，由于迭代公式中没有步长因子，而是定步长迭代，对于非二次型目标函数，有时会使函数值上升，不能保证函数值稳定地下降，甚至可能造成无法收敛的情况。因此，人们提出了“阻尼牛顿法”，即每次迭代方向仍采用$p_k$，但每次迭代需沿此方向一维搜索，寻求最优的步长因子$\lambda_k$。

#### DFP算法（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_dfp_algorithm.py)】code.newton_method.dfp_algorithm

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_dfp_algorithm.py

import numpy as np
from ..gradient_descent import golden_section_for_line_search  # code.gradient_descent.golden_section_for_line_search
from ..gradient_descent import partial_derivative  # code.gradient_descent.partial_derivative

def dfp_algorithm(func, n_features, epsilon=1e-6, distance=3, maximum=1000):
    """DFP算法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    G0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算pk
        pk = - G0 * g0

        # 一维搜索求lambda_k
        def f(x):
            """pk 方向的一维函数"""
            x2 = [x0[j] + x * float(pk[j][0]) for j in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # 更新当前点坐标
        x1 = [x0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        nabla = partial_derivative(func, x1)
        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x1

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        G1 = G0 + (dk * dk.T) / (dk.T * yk) + (G0 * yk * yk.T * G0) / (yk.T * G0 * yk)

        G0 = G1
        x0 = x1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/DFP%E7%AE%97%E6%B3%95.py)】测试

```python
>>> from code.newton_method import dfp_algorithm
>>> dfp_algorithm(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>>> dfp_algorithm(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-3.0000000003324554, -3.9999999998511546]
```

## B.4 BFGS算法

> 【补充说明】正定对称的初始矩阵$G_0$，不妨取单位矩阵$I$。

#### BFGS算法（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_bfgs_algorithm.py)】code.newton_method.bfgs_algorithm

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_bfgs_algorithm.py

import numpy as np
from ..gradient_descent import golden_section_for_line_search  # code.gradient_descent.golden_section_for_line_search
from ..gradient_descent import partial_derivative  # code.gradient_descent.partial_derivative

def bfgs_algorithm(func, n_features, epsilon=1e-6, distance=3, maximum=1000):
    """BFGS算法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    B0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)
    x0 = [0] * n_features  # 取初始点x0

    for k in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算pk
        if k == 0:
            pk = - B0 * g0  # 若numpy计算逆矩阵时有0，则对应位置会变为inf
        else:
            pk = - (B0 ** -1) * g0

        # 一维搜索求lambda_k
        def f(x):
            """pk 方向的一维函数"""
            x2 = [x0[j] + x * float(pk[j][0]) for j in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # 更新当前点坐标
        x1 = [x0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        nabla = partial_derivative(func, x1)
        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x1

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        B1 = B0 + (yk * yk.T) / (yk.T * dk) + (B0 * dk * dk.T * B0) / (dk.T * B0 * dk)

        B0 = B1
        x0 = x1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/BFGS%E7%AE%97%E6%B3%95.py)】测试

```python
>>> from code.newton_method import bfgs_algorithm
>>> bfgs_algorithm(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>>> bfgs_algorithm(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-3.0000000003324554, -3.9999999998511546]
```

#### BFGS算法(Sherman-Morrison公式)（Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_bfgs_algorithm_with_sherman_morrison.py)】code.newton_method.bfgs_algorithm_with_sherman_morrison

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_bfgs_algorithm_with_sherman_morrison.py

import numpy as np
from ..gradient_descent import golden_section_for_line_search  # code.gradient_descent.golden_section_for_line_search
from ..gradient_descent import partial_derivative  # code.gradient_descent.partial_derivative

def bfgs_algorithm_with_sherman_morrison(func, n_features, epsilon=1e-6, distance=3, maximum=1000):
    """BFGS算法(Sherman-Morrison公式)

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    I = np.identity(n_features)
    D0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算pk
        pk = - D0 * g0

        # 一维搜索求lambda_k
        def f(x):
            """pk 方向的一维函数"""
            x2 = [x0[j] + x * float(pk[j][0]) for j in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # 更新当前点坐标
        x1 = [x0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        nabla = partial_derivative(func, x1)
        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x1

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        D1 = D0 + (I - (dk * yk.T) / (yk.T * dk)) * D0 * (I - (yk * dk.T) / (yk.T * dk)) + (dk * dk.T) / (yk.T * dk)

        D0 = D1
        x0 = x1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/BFGS%E7%AE%97%E6%B3%95(Sherman-Morrison%E5%85%AC%E5%BC%8F).py)】测试

```python
>>> from code.newton_method import bfgs_algorithm_with_sherman_morrison
>>> bfgs_algorithm_with_sherman_morrison(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>>> bfgs_algorithm_with_sherman_morrison(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-3.0000000000105342, -4.000000000014043]
```

## B.5 Broyden类算法

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_broyden_algorithm.py)】code.newton_method.broyden_algorithm

```python
# https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/newton_method/_broyden_algorithm.py

import numpy as np
from ..gradient_descent import golden_section_for_line_search  # code.gradient_descent.golden_section_for_line_search
from ..gradient_descent import partial_derivative  # code.gradient_descent.partial_derivative

def broyden_algorithm(func, n_features, alpha=0.5, epsilon=1e-6, distance=3, maximum=1000):
    """Broyden算法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param alpha: [float] Broyden算法的alpha参数
    :param epsilon: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    G0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)
    I = np.identity(n_features)
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算pk
        pk = - G0 * g0

        # 一维搜索求lambda_k
        def f(x):
            """pk 方向的一维函数"""
            x2 = [x0[j] + x * float(pk[j][0]) for j in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # 更新当前点坐标
        x1 = [x0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        nabla = partial_derivative(func, x1)
        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x1

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        G_DFP = G0 + (dk * dk.T) / (dk.T * yk) + (G0 * yk * yk.T * G0) / (yk.T * G0 * yk)
        G_BFGS = G0 + (I - (dk * yk.T) / (yk.T * dk)) * G0 * (I - (yk * dk.T) / (yk.T * dk)) + (dk * dk.T) / (yk.T * dk)

        G0 = alpha * G_DFP + (1 - alpha) * G_BFGS
        x0 = x1
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E9%99%84%E5%BD%95B_%E7%89%9B%E9%A1%BF%E6%B3%95%E5%92%8C%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95/Broyden%E7%AE%97%E6%B3%95.py)】测试

```python
>>> from code.newton_method import broyden_algorithm
>>> broyden_algorithm(lambda x: x[0] ** 2, 1, epsilon=1e-6)
[0]
>>> broyden_algorithm(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6)
[-3.0000000003324554, -3.9999999998511546]
```
