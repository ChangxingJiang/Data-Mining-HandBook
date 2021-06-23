# 《统计学习方法》啃书辅助：附录C 朗格朗日对偶性

【补充说明】原始问题和对偶问题的简单理解如下：

- 原始问题：在确定 $x$ 的情况下，求使 $L(x,\alpha,\beta)$ 取得最大值的 $\alpha$ 或 $\beta$；然后在所有求出来的 $L(x,\alpha,\beta)$ 的最大值中，寻找取值最小值时的 $x$。
- 对偶问题：在确定 $\alpha$ 和 $\beta$ 的情况下，求使 $L(x,\alpha,\beta)$ 取得最小值的 $x$；然后在所有求出来的 $L(x,\alpha,\beta)$ 的最小值中，寻找取得最大值时的 $\alpha$ 和 $\beta$。

【补充说明】仿射函数：最高次数为 1 的多项式函数，一般形式为 $f(x)=Ax+b$。其中 $A$ 为 $m×n$ 矩阵，$B$ 为 $m$ 维向量。仿射函数可以将 $n$ 维向量映射到 $m$ 维向量上。

#### 为什么有时规定拉格朗日乘子大于等于 0？

简单来说，这是因为拉格朗日乘子要起到的作用决定的。

引入拉格朗日乘子是为了强制要求所有的约束条件必须被满足，当 $x$ 违反约束条件时，$L(x,\alpha,\beta) \rightarrow + \infty$，当 $x$ 满足约束条件时，$L(x,\alpha,\beta) = f(x)$。

具体地，假设 $f(x)$，$c_i(x)$，$h_j(x)$ 是定义在 $R^n$ 上的连续可微函数。考虑约束最优化问题（极大化问题可以简单地转换为极小化问题，这里仅讨论极小化问题）：

$$
\begin{align}
\min_{x \in R^n} \hspace{1em} & f(x)\\
s.t. \hspace{1em} & c_i(x) \le 0, \hspace{1em} i=1,2,\cdots,k\\
& h_j(x) = 0, \hspace{1em} j=1,2,\cdots,l
\end{align}
$$

引入拉格朗日乘子后，得到

$$
L(x,\alpha,\beta) = f(x) + \sum_{i=1}^k \alpha_i c_i (x) + \sum_{j=1}^l \beta_j h_j (x)
$$

对于约束条件 $c_i(x) \le 0$：当存在某个 $i$ 使得 $c_i(x)>0$ 时，违反了约束条件；我们需要令 $\alpha_i \rightarrow +\infty$，使 $\alpha_i c_i \rightarrow + \infty$。而对于 $c_i(x) \le 0$，满足约束条件；我们需要令当 $\alpha_i = 0$ 时，$\alpha_i c_i(x)$ 取得最大值 $0$。显然，$\alpha_i$ 不能为负数，于是我们得到了 $\alpha_i \ge 0$ 的取值范围。

对于约束条件 $h_j(x) = 0$，当存在某个 $j$ 使得 $h_j(x)>0$ 时，我们需要令 $\beta_j \rightarrow +\infty$；对于 $h_j(x)<0$，我们需要令 $\beta_j \rightarrow - \infty$；而对于 $h_j(x)=0$，满足约束条件，$\beta_j$ 取得任意值时，均有 $\beta_j h_j(x) = 0$。显然，$\beta_j$ 可以取任意值，于是我们得到了 $\beta_j \in R$ 的取值范围。

#### KKT 条件整理

1. 函数各个自变量的偏导数为 0（C.21）
2. 各个拉格朗日乘子与约束条件的乘积均为 0（C.22）
3. 各个约束条件均满足（C.23，C.25）
4. 拉格朗日乘子的定义域均满足（C.24）
