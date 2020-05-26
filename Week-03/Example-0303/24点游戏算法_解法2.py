"""
案例实现：24点游戏算法(解法2)

Version: 0.1
Author: 长行
"""

import copy
import itertools
import re
import time


def all_maybe_count_answer(n, m):
    """
    计算n和m的所有可能计算结果
    """
    answer_list = [n + m, n - m, m - n, n * m]
    if n != 0:  # 规避n为0的异常
        answer_list.append(m / n)
    if m != 0:  # 规避m为0的异常
        answer_list.append(n / m)
    return answer_list


def reverse_formula(n, m, answer):
    """
    反推由n和m计算出answer的算式

    :param n: n或计算结果为n的算式
    :param m: m或计算结果为n的算式
    :param answer: n和m计算出的结果
    """
    if re.match("^[0-9]+$", str(n)):
        fn = str(n)
    else:
        fn = "(" + str(n) + ")"
    if re.match("^[0-9]+$", str(m)):
        fm = str(m)
    else:
        fm = "(" + str(m) + ")"
    if eval(fn + "+" + fm + "==" + str(answer)):
        return str(fn) + "+" + str(fm)
    if eval(fn + "-" + fm + "==" + str(answer)):
        return str(fn) + "-" + str(fm)
    if eval(fm + "-" + fn + "==" + str(answer)):
        return str(fm) + "-" + str(fn)
    if eval(fn + "*" + fm + "==" + str(answer)):
        return str(fn) + "*" + str(fm)
    if eval(fn + "/" + fm + "==" + str(answer)):
        return str(fn) + "/" + str(fm)
    if eval(fm + "/" + fn + "==" + str(answer)):
        return str(fm) + "/" + str(fn)
    else:
        print("错误:", fm, fn, answer)
        return str(fm) + "?" + str(fn)


def get_formula(formula_list, n):
    """
    在卡牌列表formula_list中获取指定卡牌n的算式
    """
    for formula_item in formula_list:
        if formula_item[0] == n:
            return formula_item[1]
    return None


def delete_formula(formula_list, n):
    """
    从卡牌列表formula_list中删除指定卡牌n及其算式
    """
    for formula_item in formula_list:
        if formula_item[0] == n:
            formula_list.remove(formula_item)
            break


def solve(card_probability):
    card_probability = list(card_probability)
    answer = []
    for combine_1 in set(itertools.combinations(card_probability, 2)):  # 在4个数中任意抽取2个数
        for answer_1 in all_maybe_count_answer(combine_1[0], combine_1[1]):
            card_list_1 = copy.deepcopy(card_probability)
            card_list_1.remove(combine_1[0])  # 移除抽取的扑克牌
            card_list_1.remove(combine_1[1])  # 移除抽取的扑克牌
            card_list_1.append(answer_1)  # 添加抽取两张牌的计算结果
            for combine_2 in set(itertools.combinations(card_list_1, 2)):  # 在3个数中任意抽取2个数
                for answer_2 in all_maybe_count_answer(combine_2[0], combine_2[1]):
                    card_list_2 = copy.deepcopy(card_list_1)
                    card_list_2.remove(combine_2[0])  # 移除抽取的数字
                    card_list_2.remove(combine_2[1])  # 移除抽取的数字
                    card_list_2.append(answer_2)  # 添加抽取数字的计算结果
                    for combine_3 in set(itertools.combinations(card_list_2, 2)):  # 在2个数中任意抽取2个数
                        for answer_3 in all_maybe_count_answer(combine_3[0], combine_3[1]):
                            if round(answer_3, 3) == 24:
                                formula_dict = list()
                                for card in card_probability:
                                    formula_dict.append([card, str(card)])
                                formula_dict.append([answer_1, reverse_formula(get_formula(formula_dict, combine_1[0]),
                                                                               get_formula(formula_dict, combine_1[1]),
                                                                               answer_1)])
                                delete_formula(formula_dict, combine_1[0])
                                delete_formula(formula_dict, combine_1[1])
                                formula_dict.append([answer_2, reverse_formula(get_formula(formula_dict, combine_2[0]),
                                                                               get_formula(formula_dict, combine_2[1]),
                                                                               answer_2)])
                                delete_formula(formula_dict, combine_2[0])
                                delete_formula(formula_dict, combine_2[1])
                                formula = reverse_formula(get_formula(formula_dict, combine_3[0]),
                                                          get_formula(formula_dict, combine_3[1]),
                                                          answer_3)
                                answer.append(formula)
    return answer


if __name__ == "__main__":
    start_time = time.time()
    for cards in list(itertools.product([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], repeat=4))[:286]:
        solve(cards)
    print("计算时间:", 100 * (time.time() - start_time))
