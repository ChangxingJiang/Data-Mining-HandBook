from typing import Iterable
from typing import Tuple


class BinaryTreeNode:
    """二叉树结点类

    树由它的根结点表示，每个结点表示它的子树
    """

    __slots__ = "data", "left", "right"

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    @property
    def children(self) -> Iterable[Tuple]:
        """返回当前结点的非空子结点的迭代器

        子结点的返回格式为元组 (结点, 下标)
        左子结点的下标为0，右子结点的下标为1
        """
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    @property
    def is_leaf(self) -> bool:
        """当前结点是否为叶结点"""
        return (not self.data) or (all(not bool(c) for c, p in self.children))

    @property
    def height(self) -> int:
        """返回（子）树的高度

        不考虑空节点的情况
        """
        if not self:
            return 0
        elif self.is_leaf:
            return 1
        else:
            return max(c.height for c, i in self.children) + 1

    def set_child(self, index: int, child) -> None:
        """设置当前结点的指定下标的子结点的值

        如果当前结点没有该子结点，则抛出IndexError异常
        左子结点的下标为0，右子结点的下标为1
        """
        if index == 0:
            self.left = child
        elif index == 1:
            self.right = child
        else:
            raise IndexError("child index ({}) out of range".format(index))

    def get_child_pos(self, child) -> int:
        """返回当前结点指定子结点的下标

        如果当前结点没有该子结点，则返回None
        左子结点的下标为0，右子结点的下标为1
        """
        if child == self.left:
            return 0
        if child == self.right:
            return 1

    def __bool__(self):
        return self.data is not None

    def __repr__(self):
        return (self.__class__.__name__ +
                "{" + ", ".join("{}: {}".format(k, getattr(self, k)) for k in ["data", "left", "right"]) + "}")

    def __hash__(self):
        return id(self)

    # -------------------- 二叉树的遍历方法 --------------------

    def preorder(self) -> Iterable:
        """先序遍历二叉树，并返回迭代器：当前结点、当前结点的左子树、当前结点的右子树"""
        if not self:
            return
        yield self
        if self.left:
            for node in self.left.preorder():
                yield node
        if self.right:
            for node in self.right.preorder():
                yield node

    def inorder(self) -> Iterable:
        """中序遍历二叉树，并返回迭代器：当前结点的左子树、当前结点、当前结点的右子树"""
        if not self:
            return
        if self.left:
            for node in self.left.inorder():
                yield node
        yield self
        if self.right:
            for node in self.right.inorder():
                yield node

    def postorder(self) -> Iterable:
        """后序遍历二叉树，并返回迭代器：当前结点的左子树、当前结点的右子树、当前结点"""
        if not self:
            return
        if self.left:
            for x in self.left.postorder():
                yield x
        if self.right:
            for x in self.right.postorder():
                yield x
        yield self


if __name__ == "__main__":
    # Demo : children
    print(len(list(BinaryTreeNode(1).children)))  # 0
    print(len(list(BinaryTreeNode(1, left=BinaryTreeNode(2)).children)))  # 1
    print(len(list(BinaryTreeNode(1, left=BinaryTreeNode(2), right=BinaryTreeNode(3)).children)))  # 2

    # Demo : is_leaf
    print(BinaryTreeNode().is_leaf)  # True
    print(BinaryTreeNode(1, left=BinaryTreeNode(2)).is_leaf)  # False

    # Demo : height
    print(BinaryTreeNode(1).height)  # 1
    print(BinaryTreeNode(1, left=BinaryTreeNode(2)).height)  # 2
    print(BinaryTreeNode(1, left=BinaryTreeNode(2, left=BinaryTreeNode(3))).height)  # 3
