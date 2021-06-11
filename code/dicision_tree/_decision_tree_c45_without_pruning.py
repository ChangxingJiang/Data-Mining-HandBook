from ._decision_tree_id3_without_pruning import DecisionTreeID3WithoutPruning
from ._entropy import entropy


class DecisionTreeC45WithoutPruning(DecisionTreeID3WithoutPruning):
    """C4.5生成算法构造的决策树（仅支持离散型特征）-不包含剪枝"""

    def information_gain(self, x, y, idx):
        """重写计算信息增益的方法，改为计算信息增益比"""
        return super().information_gain(x, y, idx) / entropy([x[i][idx] for i in range(len(x))], base=self.base)
