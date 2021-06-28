from copy import copy


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
            # print("平方损失函数:", sum(c * c for c in r), "残差:", r)

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
