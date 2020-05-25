"""
1. 诗句数量、诗句字数是否符合绝句或律诗的要求（不考虑排律、六言的情况）
2. 诗文是否押平声韵（暂不考虑首句押韵情况），诗文的韵脚是否符合平水韵部
3. 诗句是否为拗句（是否存在孤平或拗救的情况）
4. 诗文的平仄是否符合对黏的要求

《首春》 李世民
寒随穷律变，春逐鸟声开。
平平平仄仄　平仄仄平平（十灰） —— 仄仄脚正格 平平脚正格
初风飘带柳，晚雪间花梅。
平平平仄仄　仄仄中平平（十灰） —— 【失黏】仄仄脚正格 平平脚正格
碧林青旧竹，绿沼翠新苔。
仄平平仄仄　仄仄仄平平（十灰） —— 【失黏】仄仄脚正格 平平脚正格
芝田初雁去，绮树巧莺来。
平平平仄中　仄中中平平（十灰） —— 【失黏】仄仄脚正格 平平脚正格

rhythm = 韵部
tone = 平仄
first = 出句：一联中的上一句
second = 对句：一联中的下一句
type = 原始正格
"""

import json
import re


def load_rhythm_list():
    """
    载入平水韵表并转换为dict形式
    """
    with open("平水韵表.txt", encoding="UTF-8") as file:
        rhythm_lines = file.readlines()
    rhythm_dict = dict()
    for rhythm_line in rhythm_lines:
        rhythm_name = re.search(".*(?=[平上去入]声:)", rhythm_line).group()  # 读取韵部名称
        rhythm_tune = re.search("[平上去入](?=声:)", rhythm_line).group()  # 读取韵部声调
        rhythm_characters = re.sub(".*[平上去入]声:", "", rhythm_line)  # 获取韵部包含的所有文字
        for character in rhythm_characters:
            if character not in rhythm_dict:
                rhythm_dict[character] = list()
            rhythm_dict[character].append([rhythm_name, rhythm_tune])
    return rhythm_dict


RHYTHM_LIST = load_rhythm_list()  # 导入平水韵表


def get_rhythm(character):
    """判断字所在的韵部
    若当前字不存在于平水韵表中，则返回“罕见字”；若存在于多个韵部之中，则返回所有韵部

    :param character: <str> 一个字
    :return: <str> 韵部名
    """
    rhythm_set = set()
    if character in RHYTHM_LIST:
        for rhythm_item in RHYTHM_LIST.get(character):
            rhythm_set.add(rhythm_item[0])
        if len(rhythm_set) == 1:  # 若当前字不存在于多个韵部之中
            return list(rhythm_set)[0]
        else:
            return "/".join(list(rhythm_set))
    else:
        return "罕见字"


def get_tone(character):
    """判断字所属的平仄
    若当前字不存在于平水韵表中或为无法确定平仄的多音字时，返回“中”

    :param character: <str> 一个字
    :return: <str> 平/仄/中
    """
    tone_set = set()
    if character in RHYTHM_LIST:
        for rhythm_item in RHYTHM_LIST.get(character):
            tone_set.add(re.sub("[上去入]", "仄", rhythm_item[1]))
        if len(tone_set) == 1:  # 若当前字不是多音字或是平仄相同的多音字
            return list(tone_set)[0]
        else:
            return "中"
    else:
        return "中"


def inspect_sentence_tone(sentence_tone):
    """
    判断诗句是否为拗句

    :return: <bool> 诗句是否正确, <bool> 是否需要对句救, <str> 诗句情况详细说明
    """
    if re.match("[平仄中]?[平中]?[平仄中][仄中][平中][平中][仄中]", sentence_tone):  # (仄)仄平平仄
        return True, "仄仄平平仄", "平仄脚正格"
    elif re.match("[平仄中]?[仄中]?[平仄中][平中][平中][仄中][仄中]", sentence_tone):  # (平)平平仄仄
        return True, "平平平仄仄", "仄仄脚正格"
    elif re.match("[平仄中]?[平中]?[平仄中][仄中][仄中][平中][平中]", sentence_tone):  # (仄)仄仄平平
        return True, "仄仄仄平平", "平平脚正格"
    elif re.match("[平仄中]?[仄中]?[平中][平中][仄中][仄中][平中]", sentence_tone):  # 平平仄仄平
        return True, "平平仄仄平", "仄平脚正格"
    elif re.match("[平仄中]?[平中]?[平仄中][仄中][仄中][平中][仄中]", sentence_tone):  # (仄)仄仄平仄
        return True, "仄仄平平仄", "平仄脚变格（半拗可救可不救，若救对句五言第第三字、七言第五字用平）"
    elif re.match("[平仄中]?[平中]?[平仄中][仄中][平仄中][仄中][仄中]", sentence_tone):  # (仄)仄(平)仄仄
        return True, "仄仄平平仄", "平仄脚变格（大拗必救，对句五言第第三字、七言第五字用平）"
    elif re.match("[平仄中]?[仄中]?[平中][平中][仄中][平仄中][仄中]", sentence_tone):  # 平平仄(平)仄
        return True, "平平平仄仄", "仄仄脚变格（五言第一字、七言第三字必平）"
    elif re.match("[平仄中]?[仄中]?[平仄中][仄中][平中][平中][平中]", sentence_tone):  # (仄)仄平平平
        return True, "仄仄仄平平", "平平脚变格（极其罕见）"
    elif re.match("[平仄中]?[仄中]?[仄中][平中][平中][仄中][平中]", sentence_tone):  # 仄平平仄平
        return True, "平平仄仄平", "仄平脚变格（孤平拗救，五言第一字、七言第三字必平）"
    elif re.match("[平仄中]?[仄中]?[平中][平中][平中][仄中][平中]", sentence_tone):  # 平平平仄平
        return True, "平平仄仄平", "仄平脚变格（出句拗对句救，五言第三字、七言第五字用平）"
    else:
        return False, "", "拗句"


def is_tone_same(tone_1, tone_2):
    """
    判断两个字平仄是否相同
    """
    if (tone_1 == "仄" or tone_1 == "中") and (tone_2 == "仄" or tone_2 == "中"):
        return True
    elif (tone_1 == "平" or tone_1 == "中") and (tone_2 == "平" or tone_2 == "中"):
        return True
    else:
        return False


def is_tone_differ(tone_1, tone_2):
    """
    判断两个字平仄是否不同
    :param tone_1:
    :param tone_2:
    :return:
    """
    if (tone_1 == "仄" or tone_1 == "中") and (tone_2 == "平" or tone_2 == "中"):
        return True
    elif (tone_1 == "平" or tone_1 == "中") and (tone_2 == "仄" or tone_2 == "中"):
        return True
    else:
        return False


def inspect_corresponding(first_type, second_type):
    """
    判断句子的对是否正确

    :param first_type: <str> 出句的正格
    :param second_type: <str> 对句的正格
    :return: <bool>
    """
    if len(first_type) != len(second_type):
        return False
    return is_tone_differ(first_type[-2], second_type[-2]) and is_tone_differ(first_type[-1], second_type[-1])


def inspect_sticky(last_second_type, this_first_type):
    """
    判断句子的黏是否正确

    :param last_second_type: <str> 前句对句的正格
    :param this_first_type: <str> 当前出句的正格
    :return: <bool>
    """
    if len(last_second_type) != len(this_first_type):
        return False
    return is_tone_same(last_second_type[-2], this_first_type[-2])


class Poem:
    def __init__(self, title, author, content):
        self.title = title
        self.author = author
        self.content = content.replace("\n", "").replace("\r", "")

        self.sentences = [sentence for sentence in re.split("[，。？！]", self.content) if sentence != ""]
        self.punctuations = re.findall("[，。？！]", content)

        self.is_in_rhythm = True  # 是否为近体诗
        self.not_reason = ""

        # 检查诗文的句子数量是否为绝句或律诗
        if len(self.sentences) != 4 and len(self.sentences) != 8:
            self.not_reason = "《" + self.title + "》" + self.author + " 诗句句数不是绝句或律诗"
            self.is_in_rhythm = False

        # 检查诗文中句子的字数是否为五言或七言
        if not all([len(sentence) == 5 or len(sentence) == 7 for sentence in self.sentences]):
            self.not_reason = "《" + self.title + "》 " + self.author + " 诗文中句子的字数不是五言或七言"
            self.is_in_rhythm = False

        # 计算诗句中每个字的平仄情况
        sentence_tone_list = list()
        for sentence in self.sentences:
            sentence_tone_list.append("".join([get_tone(character) for character in sentence]))

        # 判断是否押平声韵
        if not all([sentence_tone_list[i][-1] in ["平", "中"] for i in range(len(self.sentences)) if i % 2 == 1]):
            self.not_reason = "《" + self.title + "》 " + self.author + "诗文没有押韵或押仄声韵"
            self.is_in_rhythm = False

        self.output = ""
        self.output += "《" + title + "》" + author + "\n"

        last_second_type = ""

        for i in range(int(len(self.sentences) / 2)):
            first_sentence = self.sentences[2 * i + 0]  # 出句内容
            second_sentence = self.sentences[2 * i + 1]  # 对句内容

            first_tone = sentence_tone_list[2 * i + 0]  # 出句的平仄
            second_tone = sentence_tone_list[2 * i + 1]  # 对句的平仄

            second_rhythm = "（" + get_rhythm(second_sentence[-1]) + "）"  # 对句的韵脚

            first_correct, first_type, first_explanation = inspect_sentence_tone(first_tone)
            second_correct, second_type, second_explanation = inspect_sentence_tone(second_tone)

            other_analysis = ""
            if first_correct and second_correct:
                if not inspect_corresponding(first_type, second_type):  # 判断是否对
                    other_analysis += "【失对】"
                if last_second_type is not None and inspect_sticky(last_second_type, first_type):  # 判断是否黏
                    other_analysis += "【失黏】"

            last_second_type = second_type

            output_sentence = first_sentence + self.punctuations[2 * i + 0] + second_sentence + self.punctuations[
                2 * i + 1]  # 第一行输出
            output_analysis = first_tone + "　" + second_tone + second_rhythm  # 第二行输出
            output_analysis += " —— " + other_analysis + first_explanation + " " + second_explanation

            self.output += output_sentence + "\n"
            self.output += output_analysis + "\n"

    def __str__(self):
        return self.output


if __name__ == "__main__":
    # 载入整理完成的全唐诗文本语料
    with open("全唐诗.json", encoding="UTF-8") as file:
        poem_json = json.loads(file.read())

    for poem_item in poem_json["data"]:

        poem = Poem(poem_item["title"], poem_item["author"], poem_item["content"])

        if poem.is_in_rhythm:
            print(poem)
            print("点击回车继续...")
            input()
        else:
            print(poem.not_reason)
