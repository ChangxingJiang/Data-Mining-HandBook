"""
直播间弹幕数据清洗

Version: 0.1
Author: 长行
"""

import json
import re

from pyhanlp import HanLP

PUNCTUATION_LIST = [
    " ", "　", ",", "，", ".", "。", "!", "?", ";", "、", "~", "|", "·", ":", "+", "\-", "—", "*", "/", "／", "\\", "%",
    "=", "\"", "'", "（", "）", "(", ")", "\[", "\]", "【", "】", "{", "}", "《", "》", "→", "←", "↑", "↓", "↖", "↗", "↙",
    "↘", "$", "%", "_", "#", "@", "&", "√", "X", "♂", "♡", "♿", "⭐", "❤", "■", "⭕",
    "✂", "✈", "█", "ð", "▓", "ж", "⛽", "☞", "♥", "☯", "⚽", "☺", "㊙", "✨", "＊", "✌", "⚡", "⛷", "✊", "☔", "✌", "░"
]


def full_width_to_half_width(string):
    """ 将全角字符转化为半角字符
    :param string: <str> 需要转化为半角的字符串
    :return: <str> 转化完成的字符串
    """

    result = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        result += chr(inside_code)
    return result


def data_cleaning(source, del_space=False, del_t=True, del_r=True, del_n=True, simplify=True, half_width=True,
                  upper=False, lower=False, merge_3_chinese=False, merge_3_english=False, merge_3_number=False,
                  merge_3_punctuation=False):
    """
    通用自然语言文本清洗

    :param source: <str> 原始字符串
    :param del_space: <bool> 是否剔除空格(半角空格和全角空格): 默认 = False
    :param del_t: <bool> 是否剔除制表符(\t): 默认 = True
    :param del_r: <bool> 是否剔除回车符(\r): 默认 = True
    :param del_n: <bool> 是否剔除换行符(\n): 默认 = True
    :param simplify: <bool> 是否将繁体字转换为简体字: 默认=True
    :param half_width: <bool> 是否将全角字符转换为半角字符: 默认 = True
    :param upper: <bool> 是否将所有英文字母统一为大写字母(优先级低于lower): 默认 = False
    :param lower: <bool> 是否将所有英文字母统一为小写字母(优先级高于upper): 默认 = False
    :param merge_3_chinese: <bool> 是否合并连续的相同的大于等于3个中文汉字: 默认 = False
    :param merge_3_english: <bool> 是否合并连续的相同的大于等于3个英文字符: 默认 = False
    :param merge_3_number: <bool> 是否合并连续的相同的大于等于3个数字: 默认 = False
    :param merge_3_punctuation: <bool> 是否合并连续的相同的大于等于2个标点符号: 默认 = False
    :return: <str> 结果字符串
    """
    if del_space:
        source = source.replace(" ", "").replace("　", "")
    if del_t:
        source = source.replace("\t", "")
    if del_r:
        source = source.replace("\r", "")
    if del_n:
        source = source.replace("\n", "")
    if simplify:
        source = HanLP.convertToSimplifiedChinese(source)  # 使用HanLP将繁体字转换为简体字
    if half_width:
        source = full_width_to_half_width(source)  # 将全角字符转换为半角字符
    if upper:
        source = source.upper()
    if lower:
        source = source.lower()
    if merge_3_chinese:
        # 合并连续的相同的中文汉字
        for chinese_character in re.findall("([\u4e00-\u9fa5])\\1{3,}", source):
            source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
    if merge_3_english:
        # 合并连续的英文字母(将大于等于3个连续的英文字母均替换为3个)
        for chinese_character in re.findall("([A-Za-z])\\1{3,}", source):
            source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
    if merge_3_number:
        # 合并连续的数字(将大于等于3个连续的英文字母均替换为3个)
        for chinese_character in re.findall("([0-9])\\1{3,}", source):
            source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
    if merge_3_punctuation:
        punctuation_list = "".join(PUNCTUATION_LIST)
        for match_punctuation in re.findall("([" + punctuation_list + "])\\1{2,}", source):
            source = re.sub("[" + match_punctuation[0] + "]{2,}", match_punctuation * 3, source)
        source = re.sub("-{2,}", "---", source)  # 处理特殊的短横杠
    return source


if __name__ == "__main__":
    with open("聆听丶芒果鱼直播间时间切片弹幕.json", encoding="UTF-8") as file:
        barrage_json = json.loads(file.read())

    barrage_list = list()

    for barrage_item in barrage_json["data"]:
        if barrage_item[0] == "NM":
            barrage_list.append(data_cleaning(barrage_item[3], lower=True, merge_3_chinese=True, merge_3_english=True,
                                              merge_3_number=True, merge_3_punctuation=True))  # 弹幕内容

    with open("时间切片弹幕(清洗后).txt", "w+", encoding="UTF-8") as file:
        file.write("\n".join(barrage_list))
