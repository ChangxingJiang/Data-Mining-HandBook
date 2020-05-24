"""
案例实现：24点游戏算法

对于任意给定的四张扑克牌，计算是否有赢得24点游戏的方法（即使用加、减、乘、除凑成24的方法）；如果有的话，列出所有可能的方法。

解法1：枚举的方法，将所有的计算方法都枚举出来，将四张扑克牌的数字代入到所有的计算方法中，从而得到所有可能的解。

Version: 0.1
Author: 长行
"""

import itertools
import random


# def get_all_card_probability():
#     """
#     生成所有可能的24点牌组列表
#
#     :return: <list> 每个元素为一个可能的牌组（共计13^4=28561个元素）；每个元素均为一个元组，元组中的元素为牌的点数
#     """
#     probability_list = list()
#     for i in range(1, 14):
#         for j in range(1, 14):
#             for k in range(1, 14):
#                 for m in range(1, 14):
#                     probability_list.append((i, j, k, m))
#     return probability_list
#
#
# def get_one_card_probability():
#     """
#     随机生成一个24点牌组
#
#     :return: <tuple> 元组中的元素为牌的点数
#     """
#     return [random.randint(1, 13) for _ in range(4)]


if __name__ == "__main__":
    # 遍历生成所有可能的算式
    formula_list = list()
    for mark_1 in ["+", "-", "*", "/"]:
        for mark_2 in ["+", "-", "*", "/"]:
            for mark_3 in ["+", "-", "*", "/"]:
                for bracket in ["d%sd%sd%sd", "(d%sd)%sd%sd", "(d%sd%sd)%sd", "d%s(d%sd)%sd", "d%s(d%sd%sd)",
                                "(d%sd)%s(d%sd)", "d%sd%s(d%sd)"]:
                    formula_list.append((bracket % (mark_1, mark_2, mark_3)).replace("d", "%d"))

    card_probability = (3, 3, 8, 8)  # 定义需要解决的牌组

    for card_order in set(itertools.permutations(card_probability, 4)):  # 遍历所有可能的卡牌顺序（最多24种可能）
        for formula in formula_list:  # 遍历所有可能的算式（448种可能）
            final_formula = formula % card_order
            try:
                if round(eval(final_formula), 3) == 24:
                    print(final_formula)
            except ZeroDivisionError:
                continue
