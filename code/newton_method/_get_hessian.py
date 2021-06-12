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
