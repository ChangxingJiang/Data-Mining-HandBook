"""
案例实现：24点游戏算法 思路1

Version: 0.1
Author: 长行
"""

import itertools


class CardGaming:
    def __init__(self):
        self.formula_list = list()  # 存储所有可能的算式
        for marks in itertools.product(["+", "-", "*", "/"], repeat=3):
            for bracket in ["{0}%s{1}%s{2}%s{3}", "({0}%s{1})%s{2}%s{3}", "({0}%s{1}%s{2})%s{3}",
                            "{0}%s({1}%s{2})%s{3}",
                            "{0}%s({1}%s{2}%s{3})", "({0}%s{1})%s({2}%s{3})", "{0}%s{1}%s({2}%s{3})"]:
                self.formula_list.append((bracket % marks))

    def solve(self, card_probability):
        answer = []
        for card_order in set(itertools.permutations(card_probability, 4)):  # 遍历所有可能的卡牌顺序（最多24种可能）
            for formula in self.formula_list:  # 遍历所有可能的算式（448种可能）
                final_formula = formula.format(*card_order)
                try:
                    if round(eval(final_formula), 3) == 24:
                        answer.append(final_formula)
                except ZeroDivisionError:
                    continue
        return answer


if __name__ == "__main__":
    print(CardGaming().solve((3, 3, 8, 8)))
    MAYBE_CARD = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]  # 所有可能的牌
    # start_time = time.time()
    # card_gaming = CardGaming()
    # time_1 = time.time() - start_time
    # start_time = time.time()
    # for cards in list(itertools.product(MAYBE_CARD, repeat=4))[:572]:
    #     card_gaming.solve(cards)
    # print("计算时间:", time_1 + 50 * (time.time() - start_time))
