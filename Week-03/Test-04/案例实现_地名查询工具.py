"""
案例实现：地名查询工具

解析“中国地名表.json”，并实现如下功能：

* 根据给定的中国地名，判断该地名是否存在；若地名不存在，返回None；若地名存在，则给出该地名所属的上级地名和该地名包含的下级地名。例如，给出“秦皇岛市”，返回其上级地名“河北省”以及其下级地名“海港区”、“山海关区”等。
* 对于给定的中国地名A，判断该地名是否存在；若地名不存在，返回None；若地名存在，则返回该地名所属的各级上级地名。例如，给出“海港区”，返回其所有上级地名“中国”、“河北省”、“秦皇岛市”。

Version: 0.1
Author: 长行
"""

import json


class Node:  # 地名节点类
    def __init__(self, name, level, father=None):
        self._name = name
        self._level = level
        self._father = father
        self._children = []

    def add_child(self, name):  # 添加子节点
        child_node = Node(name, self._level + 1, self)
        self._children.append(child_node)
        return child_node

    def pedigree(self):  # 获取所有父节点列表(包括当前节点)
        if self.father() is not None:
            return self.father().pedigree() + [self]
        else:
            return [self]

    def father(self):
        return self._father

    def children(self):
        return self._children

    def level(self):
        return self._level

    def __str__(self):
        return self._name


class Tree:  # 地名节点树
    def __init__(self, root: str, level_grade: int):
        """
        :param root: <str> 根节点名称
        :param level_grade: <int> 节点总层级数
        """
        self.root_node = Node(root, 0)
        self.node_list = [{} for _ in range(level_grade)]
        self.node_list[0][root] = self.root_node

    def add_child(self, name: str, node: Node = None):
        if node is None:
            node = self.root_node
        child_node = node.add_child(name)
        self.node_list[node.level() + 1][name] = child_node
        return child_node

    def get_node(self, name: str, level: int = None):
        if level is None:
            for level_node_list in self.node_list:
                if name in level_node_list:
                    return level_node_list[name]
        else:
            if name in self.node_list[level]:
                return self.node_list[level][name]
        return None


def load_place():
    """
    载入中国地名表并转换为节点树的形式

    :return: <dict>省级节点; <dict> 地级节点; <dict> 县级节点
    """
    # 载入“中国地名表.json”
    with open("中国地名表.json", encoding="GBK") as file:
        place_json = json.loads(file.read())

    def add_city(city_json):  # 处理市级结构
        city_node = node_tree.add_child(city_json["name"], node=province_node)
        if "area" in city_json:
            if type(city_json["area"]) == list:  # 处理包含多个县级市的地级市
                for area in city_json["area"]:
                    node_tree.add_child(area["name"], node=city_node)
            else:  # 处理仅包含一个县级市的地级市
                node_tree.add_child(city_json["area"]["name"], node=city_node)

    # 将Json形式转换为节点树形式
    node_tree = Tree(place_json["root"]["name"], 4)  # 构建地名节点树
    for province in place_json["root"]["province"]:
        province_node = node_tree.add_child(province["name"])
        if type(province["city"]) == list:  # 处理省级行政区
            for city in province["city"]:
                add_city(city)
        else:  # 处理直辖市
            add_city(province["city"])
    return node_tree


if __name__ == "__main__":

    PLACE_TREE = load_place()  # 载入中国地名表并转换为节点树的形式

    demo_place = ["海淀区", "铜梁县", "成都市", "枣阳市", "孝感市", "绥化市", "察哈尔右翼中旗", "四川省", "宾川县", "阿拉尔市"]
    for place_name in demo_place:
        if (place_node := PLACE_TREE.get_node(place_name)) is not None:
            print(place_node, place_node.father(), str([str(child) for child in place_node.children()]))
            print(place_node, "-".join([str(node) for node in place_node.pedigree()]))
        else:
            print("地名不存在")
