import collections

from ._conditional_extropy import conditional_entropy
from ._entropy import entropy


class DecisionTreeID3:
    """ID3生成算法构造的决策树（仅支持离散型特征）"""

    class Node:
        def __init__(self, mark, ee, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点
            self.ee = ee  # 以当前结点为叶结点的经验熵

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0, alpha=0.05):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值
        self.alpha = alpha  # 决策树剪枝的参数

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成
        self._pruning(self.root)  # 决策树剪枝

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = collections.Counter(y)
        ee = entropy(y, base=self.base)  # 计算以当前结点为叶结点的经验熵

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0], ee)

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0], ee)

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], ee, use_feature=best_feature_idx)
        features = set()
        sub_x = collections.defaultdict(list)
        sub_y = collections.defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def _pruning(self, node):
        # 处理当前结点为叶结点的情况：不剪枝，直接返回
        if node.is_leaf:
            return 1, node.ee

        # 计算剪枝（以当前结点为叶结点）的损失函数
        loss1 = node.ee + 1 * self.alpha

        # 计算不剪枝的损失函数
        num, ee = 1, 0
        for child in node.children.values():
            child_num, child_ee = self._pruning(child)
            num += child_num
            ee += child_ee
        loss2 = ee + num * self.alpha

        # 处理需要剪枝的情况
        if loss1 < loss2:
            node.children = {}
            return 1, node.ee

        # 处理不需要剪枝的情况
        else:
            return num, ee

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)
