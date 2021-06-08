from abc import ABCMeta
from abc import abstractmethod


class KNNBase(metaclass=ABCMeta):
    """k近邻法的抽象基类"""

    def __init__(self, x, y, k, distance_func):
        """
        :param x: 输入变量
        :param y: 输出变量
        :param k: 最接近的结点数
        :param distance_func: 距离计算方法
        """
        self.x, self.y, self.k, self.distance_func = x, y, k, distance_func
        self._pretreatment()  # 预处理训练数据集

    def count(self, x):
        """计算实例x所属的类y"""
        nears = self._get_nearest(x)
        return self._decision_rule(nears)

    @abstractmethod
    def _decision_rule(self, nears):
        """决策规则"""

    @abstractmethod
    def _pretreatment(self):
        """预处理训练数据集"""

    @abstractmethod
    def _get_nearest(self, x):
        """寻找距离实例x最近的k个点"""
