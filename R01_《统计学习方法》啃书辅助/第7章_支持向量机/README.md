# 《统计学习方法》啃书辅助：第7章 支持向量机

**支持向量机的学习过程**：

**支持向量机的预测过程**：

**支持向量机的类别划分**：

* 用于解决二类分类问题的监督学习模型
* 非概率模型
* 线性模型&非线性模型：线性支持向量机是线性模型，核函数支持向量机是非线性模型。
* 非参数化模型
* 判别模型

**支持向量机的主要优点**：

**支持向量机的主要缺点**：

---

【补充说明】合页损失函数在P. 131有详细讲解。

#### 希尔伯特空间

希尔伯特空间，即完备正交的线性空间。“完备”是指状态的所有可能值都被这个空间包含了。例如有理数集就是不完备的。“正交”是指各个坐标基相互垂直/正交。“线性”是指不同的量之间满足加法规则。希尔伯特空间可以是无穷维，也可以是有限维。

## 7.1.1 线性可分支持向量机

#### 支持向量机与感知机的关系

![支持向量机关系图](支持向量机关系图.png)

感知机和线性可分支持向量机都假设训练数据集是线性可分的。感知机利用误分类最小的策略，求得任意一个分离超平面，此时的解有无穷多个；线性可分支持向量机则利用硬间隔最大化求最优分离超平面，此时的解是唯一的。

## 7.1.2 函数间隔和几何间隔

#### 点到超平面距离公式的推导过程

【延伸阅读】[几何间隔为什么是离超平面最近的点到超平面的距离？ - 长行的回答 - 知乎](https://www.zhihu.com/question/30217705/answer/1942943891)

## 7.1.3 间隔最大化

【补充说明】在设$x_1'$、$x_1''$、$x_2'$和$x_2''$时，要使得不等式等号成立的不等式，是约束条件中的不等式（7.14）。

## 7.1.4 学习的对偶算法

#### 为什么$w^*=0$原始最优化问题的解？

因为根据数据集的线性可分性，存在$w·x+b=0$能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即对所有$y_i=+1$的实例i，有$w·x_i+b>0$，对所有$y_i=-1$的实例i，有$w·x_i+b<0$。

用反证法，当$w=0$时，显然冲突。

## 7.2 线性支持向量机与软间隔最大化

【补充说明】本章节先给出了对偶问题的结果（7.37-7.39），然后才给出的对偶问题的推导过程。

【补充说明】KKT条件中关于$\xi$的偏导数，应写作$\nabla_\xi L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = C-\alpha_i^*-\mu_i^* = 0$。

#### 软间隔的支持向量机中样本点的位置

* 若$\alpha_i^* = 0$：则有$\mu_i=C$，$\xi_i = 0$
  * 若$y_i(w^*·x_i+b^*) > 1$。分类正确，$x_i$在间隔边界以外。
  * 若$y_i(w^*·x_i+b^*) = 1$。分类正确，$x_i$恰好落在间隔边界上。（这种情况是否能够发生？）
* 若$0 < \alpha_i^* < C$：则有$0<\mu_i<C$，$\xi_i = 0$；于是$y_i(w^*·x_i+b^*) = 1$。分类正确，$x_i$恰好落在间隔边界上。
* 若$\alpha_i^* = C$：则有$\mu_i=0$，$y_i(w^*·x_i+b^*) = 1-\xi_i$
  * 若$\xi_i = 0$：于是$y_i(w^*·x_i+b^*)-1 = 0$。分类正确，$x_i$恰好落在间隔边界上。（这种情况是否能够发生？）
  * 若$0<\xi_i<1$：于是$0<y_i(w^*·x_i+b^*)<1$。分类正确，$x_i$在间隔边界与分离超平面之间。
  * 若$\xi_i=1$：于是$y_i(w^*·x_i+b^*) = 0$。$x_i$刚好落在分离超平面上。
  * 若$\xi_i>1$：于是$y_i(w^*·x_i+b^*) < 0$。分类错误，$x_i$在分离超平面的误分类一侧。

#### 线性支持向量机（sklearn实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC7%E7%AB%A0_%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/%E7%BA%BF%E6%80%A7%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA(sklearn).py)】

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = LinearSVC()
    clf.fit(x1, y1)

    print("正确率:", clf.score(x2, y2))  # 正确率: 0.8368421052631579
```

## 7.3.1 核技巧

【重点】用线性分类方法求解非线性分类问题分为两步：首先使用一个变换将原空间的数据映射到新空间；然后在新空间里用线性分类学习方法从训练数聚中学习分类模型。核技巧就属于这样的方法。

#### 什么是核函数？

核函数隐含着一个从低维空间到高维（甚至是无穷维）空间的映射，我们几乎可以认为，这样能够将在低维中线性不可分的数据集在高维空间中找到线性可分的超平面。但是这个过程是隐式地在特征空间中进行的，不需要显式地定义特征空间和映射函数。

此外，核函数还简化了内积的计算，使得我们不需要显式地计算映射函数，而直接求出内积的值。

【延伸阅读】[机器学习有很多关于核函数的说法，核函数的定义和作用是什么？ - 文鸿郴的回答 - 知乎](https://www.zhihu.com/question/24627666/answer/28460490)

## 7.3.2 正定核

#### 正定矩阵和半正定矩阵

【参考资料】[二次型的意义是什么？有什么应用？ - 马同学的回答 - 知乎](https://www.zhihu.com/question/38902714/answer/195435181)

通过以上资料，我们可以知道对称矩阵、二次型矩阵、二次型的概念是等价的。

> **【定义】正定矩阵和负定矩阵（来自高等教育出版社的《线性代数》 P. 137）**
>
> 设二次型$f(x)=x^T A x$，如果对任何$x \ne 0$，都有$f(x)>0$（显然$f(0)=0$），则称$f$为正定二次型，并称对称矩阵$A$是正定的；如果对任何$x \ne 0$都有$f(x)<0$，则称$f$是负定二次型，并称对称矩阵$A$是负定的。

类似地，半正定矩阵定义为：如果对任何$x \ne 0$，都有$f(x)>=0$（显然$f(0)=0$），则称$f$为半正定二次型，并称对称矩阵$A$是半正定的。

在以上定义中，正定矩阵、半正定矩阵和负定矩阵都是定义在二次型的基础上的，因此一定是对称矩阵。

正定矩阵具有如下定理及推论：

> **【定理 1】（来自高等教育出版社的《线性代数》 P. 137）**
>
> $n$元二次型$f = x^T A x$为正定的充分必要条件是：它的标准型的$n$个系数全为正，即它的规范形的$n$个系数全为1，亦即它的正惯性质数等于$n$。

> **【推论】（来自高等教育出版社的《线性代数》 P. 137）**
>
> 对阵矩阵$A$为正定的充分必要条件是：$A$的特征值全为正。

> **【定理 2】（来自高等教育出版社的《线性代数》 P. 137）**
>
> 对称矩阵$A$为正定的充分必要条件是：$A$的各阶主子式都为正，即
> $$
> a_{11} > 0
> , \hspace{1em}
> \begin{vmatrix}
> a_{11} & a_{12} \\
> a_{21} & a_{22}
> \end{vmatrix} > 0
> , \hspace{1em}
> ...
> , \hspace{1em}
> \begin{vmatrix}
> a_{11} & \cdots & a_{1n} \\
> \vdots &        & \vdots \\
> a_{n1} & \cdots & a_{nn} 
> \end{vmatrix} > 0
> $$

#### 根据核函数构成希尔伯特空间的步骤解析

根据假设，我们已知 $K(x,z)$ 是定义在 $X×X$ 上的对称函数，并且对任意的 $x_1,x_2,\cdots,x_m \in X$，$K(x,z)$ 关于 $x_1,x_2,\cdots,x_m$ 的 Gram 矩阵是半正定的。

##### 1. 定义映射，构成向量空间 S

先定义映射

$$
\phi : x \rightarrow K(·,x) \tag{1}
$$

> 根据核函数的定义可知，核函数定义为 $K(x,z) = \phi(x)·\phi(z)$，从输入空间到特征空间的映射函数可以显式地表达为 $\phi : x \rightarrow \phi(x)$。但是在核技巧下，我们并不需要显式地定义映射函数，因此采用上式（1）隐式地定义映射函数。
>
> 为方便理解，也可以将上式不严谨地理解为：只代入了一个参数 $\phi(x)$ 的核函数，而 $·$ 表示尚未代入的另一个参数。

根据这一映射，对任意 $x_i \in X$，$\alpha_i \in R$，$i=1,2,\cdots,m$，定义线性组合

$$
f(·)=\sum_{i=1}^m \alpha_i K(·,x_i) \tag{2}
$$

> 于是，上式（2）就是对只代入一个参数 $\phi(x)$ 的核函数的线性变换结果（加法运算和数乘运算）。
>
> 为方便理解，也可以将上式不严谨地理解为：只代入了一个参数 $\phi(\sum_{i=1}^M \alpha_i x_i)$ 的核函数。
>
> 之所以要定义这个线性组合，是因为因为向量空间是带有加法和标量乘法的集合。通过线性组合的方式令包含的元素对加法和数乘运算封闭，从而使 $S$ 成为向量空间。

考虑由线性组合为元素的集合 S。由于集合 S 对加法和数乘运算是封闭的，所以 S 构成一个向量空间。

##### 2. 在 S 上定义内积，使其成为内积空间

在 S 上定义一个运算\*；对任意 $f,g \in S$，

$$
f(·)=\sum_{i=1}^m \alpha_i K(·,x_i) \tag{3}
$$

$$
g(·)=\sum_{j=1}^l \beta_j K(·,z_j) \tag{4}
$$

> 同样地，$f(·)$ 和 $g(·)$ 也可以分别理解为各代入了一个参数的核函数的线性变换结果；同时也可以理解为只代入了一个参数 $\phi(\sum_{i=1}^M \alpha_i x_i)$ 的核函数和只代入了一个参数 $\phi(\sum_{j=1}^M \beta_i z_j)$ 的核函数。其中 $·$ 表示另一个尚未被代入的参数，这个参数应该是也是一个带入了一个参数的核函数或它的线性变换结果。

定义运算 $*$

$$
f * g = \sum_{i=1}^m \sum_{j=1}^l \alpha_i \beta_i K(x_i,z_j) \tag{5}
$$

证明运算\*是空间 $S$ 的内积。为此要证：

$$
\begin{align}
1. \hspace{1em} & (cf) * g = c (f*g),c\in R \tag{6} \\
2. \hspace{1em} & (f+g)*h = f*h + g*h,h \in S \tag{7} \\
3. \hspace{1em} & f*g = g*f \tag{8} \\
4. \hspace{1em} & f*f \ge 0, \tag{9} \\
& f*f=0 \Leftrightarrow f=0 \tag{10}
\end{align}
$$

> 以上四个条件即内积的四个运算性质，若运算 $*$ 能够满足内积的所有运算性质，则可以认为运算 $*$ 是空间 $S$ 的内积。
>
> $f=0$ 的含义是：当 $f=0$ 时，对任意的 $x$ 都有 $f(·)=\sum_{i=1}^m \alpha_i K(·,x_i)=0$。
>
> $f*f=0$ 的含义是：当 $f*f=0$ 时，对任意的 $x$ 都有 $f*f = \sum_{i,j=1}^m \alpha_i \alpha_j K(x_i,x_j) = 0$。

其中，（6）~（8）由式（2）-式（4）及 $K(x,z)$ 的对称性容易得到。

> （6）和（7）的证明直接将 $c \in R$、$h \in S$ 代入到式（2）及式（5）中即可；（8）的证明使用 $K(x,z)$ 是对称函数的性质即可。

现证（9）。由式（2）及式（5）可得：

$$
f * f = \sum_{i,j=1}^m \alpha_i \alpha_j K(x_i,x_j) \tag{11}
$$

由 Gram 矩阵的半正定性知上式右端非负，即 $f * f \ge 0$。

> 根据假设，$K(x,z)$ 关于 $x_1,x_2,\cdots,x_m$ 的 Gram 矩阵，即 $\begin{bmatrix}K(x_i,x_j)\end{bmatrix}_{m×m}$，是半正定矩阵，于是有 $x^T (\begin{bmatrix}K(x_i,x_j)\end{bmatrix}_{m×m}) x \ge 0$。
>
> 令 $x = \begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_m \end{bmatrix}^T$，则有
>
> $$
> \begin{bmatrix}
> \alpha_1 & \alpha_2 & \cdots & \alpha_m
> \end{bmatrix}
> 
> \begin{bmatrix}
> K(x_1,x_1) & K(x_1,x_2) & \cdots & K(x_1,x_m) \\
> K(x_2,x_1) & K(x_2,x_2) & \cdots & K(x_2,x_m) \\
> \vdots & \vdots & \ddots & \vdots \\
> K(x_m,x_1) & K(x_m,x_2) & \cdots & K(x_m,x_m)
> \end{bmatrix}
> 
> \begin{bmatrix}
> \alpha_1 \\
> \alpha_2 \\
> \vdots \\
> \alpha_m
> \end{bmatrix}
> 
> \ge 0
> $$
>
> 解得 $\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j K(x_i,x_j) \ge 0$，即上式（11）的右端。

再证（10）。充分性显然。

> 充分性：若 $f=0$，则 $f*f=0$。当 $f=0$ 时，有 $\alpha_i=0$，于是 $f*f$ 显然为 0。
>
> 必要性：若 $f*f=0$，则 $f=0$。

为证必要性，首先证明不等式：

$$
|f*g|^2 \le (f*f)(g*g) \tag{12}
$$

设 $f,g \in S$，$\lambda \in R$，则 $f + \lambda g \in S$，于是，

$$
(f+\lambda g)*(f+\lambda g) \ge 0
$$

$$
f*f + 2 \lambda (f*g) + \lambda^2 (g*g) \ge 0
$$

> 利用 $S$ 是对加法和数乘运算封闭的向量空间，以及已经证明的式（6）-（8）。

其左端是 $\lambda$ 的二次三项式，非负，其判别式小于等于 0，即

$$
(f*g)^2-(f*f)(g*g) \le 0
$$

于是不等式（12）得证。现证若 $f*f=0$，则 $f=0$。事实上，若

$$
f(·)=\sum_{i=1}^m \alpha_i K(·,x_i)
$$

则运算 $*$ 的定义式（5），对任意的 $x \in X$，有

$$
K(·,x)*f=\sum_{i=1}^m \alpha_i K(x,x_i) = f(x)
$$

> $K(·,x)$ 是一个没有经过线性变换的，只代入了一个参数 $\phi(x)$ 的核函数。根据式（3）和定义式（5）即可得出。

于是，

$$
|f(x)|^2 = |K(·,x)*f|^2 \tag{13}
$$

由（12）和（9）有

$$
\begin{align}
|K(·,x)*f|^2 & \le (K(·,x)*K(·,x))(f*f) \\
& = K(x,x)(f*f)
\end{align}
$$

由（13）有

$$
|f(x)|^2 \le K(x,x)(f*f)
$$

此式表明，当 $f*f=0$ 时，对任意的 $x$ 都有 $|f(x)|=0$。

至此，证明了 $*$ 为向量空间 $S$ 的内积。赋予内积的向量空间为内积空间。因此 $S$ 是一个内积空间。

> 内积空间是带有内积的向量空间。因为增加了内积，所以允许我们在内积空间中讨论向量的角度和长度。

既然 $*$ 为 $S$ 的内积运算，那么仍然用·表示，即若

$$
f(·)=\sum_{i=1}^m \alpha_i K(·,x_i), \hspace{1em} g(·)=\sum_{i=1}^l \beta_j K(·,z_j)
$$

则

$$
f·g = \sum_{i=1}^m \sum_{j=1}^l \alpha_i \beta_i K(x_i,z_j) \tag{14}
$$

##### 3. 将内积空间 S 完备化为希尔伯特空间

> 希尔伯特空间是完备化的内积空间。

现在将内积空间 $S$ 完备化。由式（14）定义的内积可以得到范数

$$
||f||=\sqrt{f·f} \tag{15}
$$

因此，$S$ 是一个赋范向量空间。根据泛函分析理论，对于不完备的赋函向量空间 $S$，一定可以使之完备化，得到完备的赋范向量空间 $H$。一个内积空间，当作为一个赋范向量空间是完备的时候，就是希尔伯特空间。这样，就得到了希尔伯特空间 $H$。

这一希尔伯特空间 $H$ 称为再生核希尔伯特空间。这是由于核 $K$ 具有再生性，即满足

$$
K(·,x)·f=f(x) \tag{16}
$$

及

$$
K(·,x)·(·,z)=K(x,z) \tag{17}
$$

称为再生核。

> 以上两式（16）和（17）可通过式（3）和（5）得出。

#### 正定核的对称性证明

因为内积满足交换律，于是有
$$
K(x,z) = \phi(x)·\phi(z) = \phi(z)·\phi(x) = K(z,x)
$$
得证。

#### 正定核充要条件必要性证明

> 对最后一步略作修改，使其更容易理解。更新内容如下：

对任意$m$维向量$C = (c_1,c_2,\cdots,c_m) \in R^m$，有
$$
\begin{align}
C^T \ [K(x_i,j_i)]_{m×m} \ C
& = \sum_{i=1}^m \sum_{j=1}^m c_i c_j K(x_i,x_j) \\
& = \sum_{i=1}^m \sum_{j=1}^m c_i c_j (\phi(x_i) · \phi(y_i)) \\
& = \Big{(} \sum_{i=1}^m c_i \phi(x_i) \Big{)} · \Big{(} \sum_{j=1}^m c_j \phi(x_j) \Big{)} \\
& = \Bigg{|}\Bigg{|} \sum_{i=1}^m c_i \phi(x_i) \Bigg{|}\Bigg{|} ^2 \ge 0
\end{align}
$$
因为$[K(x_i,j_i)]_{m×m}$为对称矩阵，故$[K(x_i,j_i)]_{m×m}$为半正定矩阵。

#### 正定核和Mercer核的区别？

> **【定义 1】Mercer核的等价定义（来自《数据挖掘中的新方法——支持向量机》P. 111）**
>
> 称函数$K(x,x')$为Mercer核，如果$K(x,x')$是定义在$X×X$上的连续对称函数，其中$X$是$R^n$上的紧集，且$K(x,x')$关于任意的$x_1,\cdots,x_l \in X$的Gram矩阵半正定。

> **【定义 2】正定核的等价定义（来自《数据挖掘中的新方法——支持向量机》P. 116）**
>
> 设$X$是$R^n$的子集。称定义在$X×X$上的对称函数$K(x,x')$为一个正定核，如果对任意的$x_1,\cdots,x_l \in X$，$K(·,·')$相对于$x_1,\cdots,x_l$的Gram矩阵都是半正定的。

虽然两个定义的表述略有差异，但是我们可以发现正定核与Mercer核的区别如下：

* 正定核函数是定义在$X×X$上的连续对称函数；而Mercer核函数是定义在$X×X$上的连续对称函数；
* 正定核函数要求$X$是$R^n$的子集即可；而Mercer核函数要求$X$是$R^n$上的紧集。

## 7.3.3 常用核函数

> 【补充说明】书中所描述的子串的概念，类似于通常所用的子序列的概念。通常子串是指连续的子序列，即书中拥有连续的$i$的子串。

#### 其他常用核函数

线性核函数：
$$
K(x,z) = x·z+c
$$
幂质数核函数：
$$
K(x,z) = exp(-\frac{||x-z||}{2 \sigma^2})
$$
拉普拉斯核函数：
$$
K(x,z) = exp(-\frac{||x-z||}{\sigma})
$$

#### 余弦相似度

余弦相似度的计算公式如下：
$$
similarity = cos(\theta) = \frac{A·B}{||A|| \ ||B||} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n (A_i)^2} × \sqrt{\sum_{i=1}^n (B_i)^2}}
$$
其中$A_i$和$B_i$分别为向量$A$和$B$中的特征。

#### 字符串核函数的动态规划计算

已知两个字符串 $s$ 和 $t$ 上的字符串核函数是基于映射 $\phi_n$ 的特征空间中的内积，即字符串 $s$ 和 $t$ 中长度等于 $n$ 的所有子串组成的特征向量的余弦相似度。其计算公式如下：

$$
\begin{align*}
k_n(s,t) & =\sum_{u \in \Sigma^n} [\phi_n(s)]_u [\phi_n(t)]_u \\
& =\sum_{u \in \Sigma^n} \sum_{(i,j):s(i)=t(j)=u} \lambda^{l(i)} \lambda^{l(j)}
\end{align*}
$$

其中 $\lambda$ 是衰减参数，$u$ 是相同子序列，$l(i)$ 为子序列在 $s$ 中的长度（最后 1 个字符的下标-第 1 个字符的下标+1），$l(j)$ 为子序列在 $t$ 中的长度。

观察上式，我们可以发现，每一个相同子串所提供的相似度，都是 $\lambda^{l(i)+l(j)}$，即衰减参数的两个子串的长度之和次方；于是有长度每增加 $1$，所提供的相似度都只需要乘以 $\lambda$ 即可。

于是，我们很自然地想到，可以用类似计数 DP 的方法，当前状态（$s1$ 的前 $i$ 个字符和 $s2$ 的前 $j$ 个字符），只需要记录所有长度为 $l$ 为相同子序列提供的相似度之和即可。当考虑下一个状态（例如 $s1$ 的前 $i+1$ 个字符和 $s2$ 的前 $j$ 个字符）时，只需要简单地将上一个状态乘以衰减因子 $\lambda$ 即可。需要注意的是，我们需要考虑可能存在的被重复计算的问题。

具体地，状态矩阵如下：

`dp[l][i]][j]`：$s1$ 的前 $i$ 个字符和 $s2$ 的前 $j$ 个字符中，所有长度为 $l$ 的相同子序列所提供的相似度的和。因为考虑到状态转移中，只会用到 $l$ 和 $l-1$，所以可以省略 $l$ 以节约空间。

状态转移方程如下（其中`att`为衰减因子）：

`dp[l][i][j] = dp[l][i-1][j] * att + dp[l][i][j-1] * att + dp[l][i-1][j-1] * att * att`（第 3 项是在处理被重复计算的问题）

在当前字符相同时，额外增加使用当前相同字符的情况：`dp[l][i][j] += dp[l-1][i-1][j-1] * att * att`；同时，当前子序列长度的核函数的结果由这部分累加即可。

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC7%E7%AB%A0_%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%A0%B8%E5%87%BD%E6%95%B0%E7%9A%84%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E8%AE%A1%E7%AE%97.py)】

```python
def count_kernel_function_for_string(s1, s2, length: int, att: float):
    """计算字串长度等于n的字符串核函数的值

    :param s1: 第1个字符串
    :param s2: 第2个字符串
    :param length: 需要查找的英文字符集长度
    :param att: 衰减参数
    :return: 字符串核函数的值
    """

    # 计算字符串长度
    n1, n2 = len(s1), len(s2)

    # 定义状态矩阵
    dp1 = [[1] * (n2 + 1) for _ in range(n1 + 1)]

    # 字符串的核函数的值的列表：ans[i] = 子串长度为(i+1)的字符串核函数的值
    ans = []

    # 依据子串长度进行状态转移：[1,n]
    for l in range(length):
        dp2 = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        # 定义当前子串长度的核函数的值
        res = 0

        # 进行状态转移
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                dp2[i][j] += dp2[i - 1][j] * att + dp2[i][j - 1] * att - dp2[i - 1][j - 1] * att * att
                if s1[i - 1] == s2[j - 1]:
                    dp2[i][j] += dp1[i - 1][j - 1] * att * att
                    res += dp1[i - 1][j - 1] * att * att  # 累加当前长度核函数的值

        dp1 = dp2
        ans.append(res)

    return ans[-1]
```

时间复杂度、空间复杂度分析：

- 时间复杂度：$O(N1×N2×Length)$；其中 $N1$ 为第 1 个字符串的长度，$N2$ 为第 2 个字符串的长度，$Length$ 为子串长度
- 空间复杂度：$O(N1×N2)$

#### 非线性支持向量机（sklearn实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC7%E7%AB%A0_%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA(sklearn).py)】

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = SVC(kernel="rbf")
    clf.fit(x1, y1)

    print("正确率:",clf.score(x2, y2))  # 正确率: 0.9263157894736842
```

## 7.4 序列最小最优化算法

【补充说明】书中的$\alpha_1=-y_1 \sum_{i=2}^N \alpha_i y_i$有误，正确的应为$\alpha_1=- \frac{\sum_{i=2}^N \alpha_i y_i}{y_1}$。但在语境下，假设$\alpha1$和$\alpha_2$为两个变量，其他固定，因此，语境下应该是$\alpha_1 y_1 + \alpha_2 y_2 = - \sum_{i=3}^N y_i \alpha_i$。

#### 启发式算法

启发式算法通过基于直观或经验构造的算法，在可接受的花费下给出待解决组合优化问题的一个可行解，该可行解与最优解的偏离程度一般不能被估计。目前常用的启发式算法包括模拟退火算法、遗传算法、蚁群算法、人工神经网络算法。

#### 为什么$(\alpha_1,\alpha_2)$平行于盒子的对角线？

当$y1 \ne y2$时，有$y1=-y2$，代入式7.102有$y_1 \alpha_1 - y_1 \alpha_2 = k$，于是有$\alpha_1 - \alpha_2 = \frac{k}{y1}$。

当$y_1=y_2$时，有$y1=y2$，代入式7.102有$y_1 \alpha_1 + y_1 \alpha_2 = k$，于是有$\alpha_1 + \alpha_2 = \frac{k}{y1}$。

#### 计算L和H

如果$y1 \ne y2$，则有$\alpha_1 - \alpha_2 = \frac{k}{y1} = (\alpha_1^{old} - \alpha_2^{old})$，于是有$\alpha_2 = \alpha_1 + (\alpha_2^{old} - \alpha_1^{old})$。代入$\alpha_1$的最小值$\alpha_1=0$，得$\alpha_2$的最小值$\alpha_2 = \alpha_2^{old} - \alpha_1^{old}$；代入$\alpha_1$的最小值$\alpha_1=C$，得到$\alpha_2$的最大值$\alpha_2 = C+\alpha_2^{old} - \alpha_1^{old}$。

如果$y1= y2$，则有$\alpha_1 + \alpha_2 = \frac{k}{y1} = (\alpha_1^{old} + \alpha_2^{old})$，于是有$\alpha_2 = -\alpha_1 + (\alpha_1^{old} + \alpha_2^{old})$。代入$\alpha_1$的最大值$\alpha_1=C$，得$\alpha_2$的最小值$\alpha_2 = \alpha_1^{old} + \alpha_2^{old} - C$；代入$\alpha_1$的最小值$\alpha_1 = 0$，得$\alpha_2$的最大值$\alpha_2 = \alpha_1^{old} + \alpha_2^{old}$。

#### $g(x)$和$E_i$的含义

$g(x)$为当前模型对输入$x$的预测值；

$E_i$为当前模型对输入$x_i$的预测值与真实值之间的有符号的差。

#### SMO实现的支持向量机（原生Python实现）

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/code/svm/_svm.py)】code.svm.SVM

```python
import numpy as np

class SVM:
    """支持向量机

    :param X: 输入变量列表
    :param Y: 输出变量列表
    :param C: 正则化项（惩罚参数：C越大，对误分类的惩罚越大）
    :param kernel_func: 核函数
    :param tol: 容差
    :param max_iter: 最大迭代次数
    """

    def __init__(self, X, Y, kernel_func=None, C=1, tol=1e-4, max_iter=100):
        # ---------- 检查参数 ----------
        # 检查输入变量和输出变量
        if len(X) != len(Y):
            raise ValueError("输入变量和输出变量的样本数不同")
        if len(X) == 0:
            raise ValueError("输入样本数不能为0")
        self.X, self.Y = X, Y

        # 检查正则化项
        if C <= 0:
            raise ValueError("正则化项必须严格大于0")
        self.C = C

        # 检查核函数
        if kernel_func is None:
            kernel_func = self._linear_kernel  # 当未设置核函数时默认使用线性核函数
        self.kernel_func = kernel_func

        # 检查容差
        if tol <= 0:
            raise ValueError("容差必须大于0")
        self.tol = tol

        # 检查最大迭代步数
        if max_iter <= 0:
            raise ValueError("迭代步数必须大于0")
        self.max_iter = max_iter

        # ---------- 初始化计算 ----------
        self.n_samples = len(X)  # 计算样本数
        self.n_features = len(X[0])  # 计算特征数
        self.kernel_matrix = self._count_kernel_matrix()  # 计算核矩阵

        # ---------- 取初值 ----------
        self.A = np.zeros(self.n_samples)  # 拉格朗日乘子(alpha)
        self.b = 0  # 参数b
        self.E = [float(-self.Y[i]) for i in range(self.n_samples)]  # 初始化Ei的列表

        # ---------- SMO算法训练支持向量机 ----------
        self.smo()  # SMO算法计算了拉格朗日乘子的近似解
        self.support = [i for i, v in enumerate(self.A) if v > 0]  # 计算支持向量的下标列表

    def smo(self):
        """使用序列最小最优化(SMO)算法训练支持向量机"""
        for k in range(self.max_iter):
            change_num = 0  # 更新的样本数

            for i1 in self.outer_circle():  # 外层循环：依据7.4.2.1选择第1个变量（找到a1并更新后继续向后遍历，而不回到第1个）
                i2 = next(self.inner_circle(i1))  # 内层循环：依据7.4.2.2选择第2个变量（没有处理特殊情况下用启发式规则继续寻找a2）

                a1_old, a2_old = self.A[i1], self.A[i2]
                y1, y2 = self.Y[i1], self.Y[i2]
                k11, k22, k12 = self.kernel_matrix[i1][i1], self.kernel_matrix[i2][i2], self.kernel_matrix[i1][i2]

                eta = k11 + k22 - 2 * k12  # 根据式(7.107)计算η(eta)
                a2_new = a2_old + y2 * (self.E[i1] - self.E[i2]) / eta  # 依据式(7.106)计算未经剪辑的a2_new

                # 计算a2_new所在对角线线段端点的界
                if y1 != y2:
                    l = max(0, a2_old - a1_old)
                    h = min(self.C, self.C + a2_old - a1_old)
                else:
                    l = max(0, a2_old + a1_old - self.C)
                    h = min(self.C, a2_old + a1_old)

                # 依据式(7.108)剪辑a2_new
                if a2_new > h:
                    a2_new = h
                if a2_new < l:
                    a2_new = l

                # 依据式(7.109)计算a_new
                a1_new = a1_old + y1 * y2 * (a2_old - a2_new)

                # 依据式(7.115)和式(7.116)计算b1_new和b2_new并更新b
                b1_new = -self.E[i1] - y1 * k11 * (a1_new - a1_old) - y2 * k12 * (a2_new - a2_old) + self.b
                b2_new = -self.E[i2] - y1 * k12 * (a1_new - a1_old) - y2 * k22 * (a2_new - a2_old) + self.b
                if 0 < a1_new < self.C and 0 < a2_new < self.C:
                    self.b = b1_new
                else:
                    self.b = (b1_new + b2_new) / 2

                # 更新a1,a2
                self.A[i1], self.A[i2] = a1_new, a2_new

                # 依据式(7.105)计算并更新E
                self.E[i1], self.E[i2] = self._count_g(i1) - y1, self._count_g(i2) - y2

                if abs(a2_new - a2_old) > self.tol:
                    change_num += 1

            print("迭代次数:", k, "change_num =", change_num)

            if change_num == 0:
                break

    def predict(self, x):
        """预测实例"""
        return np.sign(sum(self.A[i] * self.Y[i] * self.kernel_func(x, self.X[i]) for i in self.support) + self.b)

    def _linear_kernel(self, x1, x2):
        """计算特征向量x1和特征向量x2的线性核函数的值"""
        return sum(x1[i] * x2[i] for i in range(self.n_features))

    def outer_circle(self):
        """外层循环生成器"""
        for i1 in range(self.n_samples):  # 先遍历所有在间隔边界上的支持向量点
            if -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1
        for i1 in range(self.n_samples):  # 再遍历整个训练集的所有样本点
            if not -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1

    def inner_circle(self, i1):
        """内层循环生成器：未考虑特殊情况下启发式选择a2的情况"""
        max_differ = 0
        i2 = -1
        for ii2 in range(self.n_samples):
            differ = abs(self.E[i1] - self.E[ii2])
            if differ > max_differ:
                i2, max_differ = ii2, differ
        yield i2

    def _count_kernel_matrix(self):
        """计算核矩阵"""
        kernel_matrix = [[0] * self.n_samples for _ in range(self.n_samples)]
        for i1 in range(self.n_samples):
            for i2 in range(i1, self.n_samples):
                kernel_matrix[i1][i2] = kernel_matrix[i2][i1] = self.kernel_func(self.X[i1], self.X[i2])
        return kernel_matrix

    def _count_g(self, i1):
        """依据式(7.104)计算g(x)"""
        return sum(self.A[i2] * self.Y[i2] * self.kernel_matrix[i1][i2] for i2 in range(self.n_samples)) + self.b

    def _satisfied_kkt(self, i):
        """判断是否满足KKT条件"""
        ygi = self.Y[i] * self._count_g(i)  # 计算 yi*g(xi)
        if -self.tol < self.A[i] < self.tol and ygi >= 1 - self.tol:
            return True  # (7.111)式的情况: ai=0 && yi*g(xi)>=1
        elif -self.tol < self.A[i] < self.C + self.tol and abs(ygi - 1) < self.tol:
            return True  # (7.112)式的情况: 0<ai<C && yi*g(xi)=1
        elif self.C - self.tol < self.A[i] < self.C + self.tol and ygi <= 1 + self.tol:
            return True  # (7.113)式的情况: ai=C && yi*g(xi)<=1
        else:
            return False
```

【[源码地址](https://github.com/ChangxingJiang/Data-Mining-HandBook/blob/master/R01_%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B%E5%95%83%E4%B9%A6%E8%BE%85%E5%8A%A9/%E7%AC%AC7%E7%AB%A0_%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/SMO%E5%AE%9E%E7%8E%B0%E7%9A%84%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA(%E5%8E%9F%E7%94%9FPython).py)】测试

```python
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from code.svm import SVM

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    start_time = time.time()

    svm = SVM(x1, y1)
    n1, n2 = 0, 0
    for xx, yy in zip(x2, y2):
        if svm.predict(xx) == yy:
            n1 += 1
        else:
            n2 += 1

    end_time = time.time()

    print("正确率:", n1 / (n1 + n2))  # 正确率: 0.9526315789473684
    print("运行时间:", end_time - start_time)  # 运行时间: 23.218594551086426
```









