"""
案例实现：24点游戏算法

对于任意给定的四张扑克牌，计算是否有赢得24点游戏的方法（即使用加、减、乘、除凑成24的方法）；如果有的话，列出所有可能的方法。

解法2：类似于人们思考的逻辑。我们先从A1、A2、A3、A4四张扑克牌中任意抽取两张进行四则运算得到结果B1；接着，从B1和之前没使用的两张扑克牌B2、B3中再任意抽取两个数进行四则运算得到结果C1；最后，从C1和上一轮运算中没有抽到的数C2进行四则运算，得到最终结果D，如果结果D为24则该解正确。

Version: 0.1
Author: 长行
"""

import itertools

if __name__ == "__main__":
    card_probability = (3, 3, 8, 8)  # 定义需要解决的牌组

    for combine_1 in set(itertools.permutations(card_probability, 2)):  # 在A1、A2、A3、A4四张扑克牌中任意抽取两张
        for mark_1 in ["%f+%f", "%f-%f", "%f*%f", "%f/%f"]:
            try:
                card_list_1 = list(card_probability)
                card_list_1.remove(combine_1[0])  # 移除抽取的扑克牌
                card_list_1.remove(combine_1[1])  # 移除抽取的扑克牌
                card_list_1.append(eval(mark_1 % combine_1))  # 添加抽取两张牌的计算结果
                card_probability_1 = tuple(card_list_1)
                for combine_2 in set(itertools.permutations(card_probability_1, 2)):  # 在B1、B2、B3中抽取任意两个数
                    for mark_2 in ["%f+%f", "%f-%f", "%f*%f", "%f/%f"]:
                        try:
                            card_list_2 = list(card_probability_1)
                            card_list_2.remove(combine_2[0])  # 移除抽取的数字
                            card_list_2.remove(combine_2[1])  # 移除抽取的数字
                            card_list_2.append(eval(mark_2 % combine_2))  # 添加抽取数字的计算结果
                            card_probability_2 = tuple(card_list_2)
                            for combine_3 in set(itertools.permutations(card_probability_2, 2)):  # 在B1、B2、B3中抽取任意两个数
                                for mark_3 in ["%f+%f", "%f-%f", "%f*%f", "%f/%f"]:
                                    try:
                                        result = eval(mark_3 % combine_3)  # 计算最终结果
                                        if round(result, 3) == 24:
                                            print(mark_1 % combine_1, "→", mark_2 % combine_2, "→", mark_3 % combine_3)
                                    except ZeroDivisionError:
                                        continue
                        except ZeroDivisionError:
                            continue
            except ZeroDivisionError:
                continue
