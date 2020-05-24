# Day08 : 基于HanLP实现的中文文本清洗

> **作者**：长行
>
> **时间**：2020.05.10

> 这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

在拿到自然语言语料之后，并不能直接用来分析，需要先进行数据清洗的工作。

## 英文语料数据清洗

通常而言，英文语料的清洗包括如下步骤：

1. 文本规范化（Normalization），将文本中所有的字母统一转换为大写或小写，如不需要标点符号也可以过滤掉文本中的标点符号。
2. 英文语料分词（Tokenization），将文本拆分为无法再分拆的符号。
3. 去除停用词（Stop Word），去除文本中没有意义的停用词
4. 变形标准化（Stemming and Lemmatization），将复数、过去式、动名词等还原。

## 中文语料数据清洗

相较于英文语料，中文文本规范化需要额外处理繁简转换和全角/半角的问题。

因此，中文语料的情绪大体包括如下步骤：

1. 文本规范化（Normalization）：将全角字符转换为半角字符、将繁体字转换为简体字、将中文语料中包含的英文大小写统一
2. 中文分词（Tokenization）
3. 去除停用词（Stop Word）
4. 变形标准化（Stemming and Lemmatization）

其中，中文分词和去除停用词在之前（第7天）的学习中已经掌握。

## 中文文本规范化

中文语料文本规范化可以通过HanLP中的CharTable实现。

```python
from pyhanlp import *

if __name__ == "__main__":
    CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    print(CharTable.convert('ｄｅａｄｌｉｎｄ並沒有提高你的效率，只是給了你交一堆Ｔｒａｓｈ上去的勇氣'))
```

运行结果

deadlind并没有提高你的效率,只是给了你交一堆trash上去的勇气

## 中文变形词标准化

变形标准化在不同的应用场景中各不相同，例如，我在分析直播间弹幕时，就出现了大量的不同长度的“IGGGGGG”、“爬爬爬”、“!!!”等，因此我将超过2个或3个的相同数字、标点符号、英文字母或汉字统一替换为2个或3个，Python实现代码如下：

```python
import re

# 标点符号/特殊符号词典
PUNCTUATION_LIST = [
    " ", "　", ",", "，", ".", "。", "!", "?", ";", "、", "~", "|", "·", ":", "+", "\-", "—", "*", "/", "／", "\\", "%",
    "=", "\"", "'", "（", "）", "(", ")", "\[", "\]", "【", "】", "{", "}", "《", "》", "→", "←", "↑", "↓", "↖", "↗", "↙",
    "↘", "$", "%", "_", "#", "@", "&", "√", "X", "♂", "♡", "♿", "⭐", "❤", "■", "⭕",
    "✂", "✈", "█", "ð", "▓", "ж", "⛽", "☞", "♥", "☯", "⚽", "☺", "㊙", "✨", "＊", "✌", "⚡", "⛷", "✊", "☔", "✌", "░"
]


def stemming(string):
    # 将大于等于2个连续的相同标点符号均替换为1个
    punctuation_list = "".join(PUNCTUATION_LIST)
    for match_punctuation in re.findall("([" + punctuation_list + "])\\1{2,}", string):
        string = re.sub("[" + match_punctuation + "]{2,}", match_punctuation * 3, string)
    string = re.sub("-{2,}", "---", string)  # 处理特殊的短横杠

    # 将大于等于3个连续的中文汉字均替换为3个
    for chinese_character in re.findall("([\u4e00-\u9fa5])\\1{3,}", string):
        string = re.sub("[" + chinese_character + "]{3,}", chinese_character * 3, string)

    # 将大于等于3个连续的英文字母均替换为3个
    for chinese_character in re.findall("([A-Za-z])\\1{3,}", string):
        string = re.sub("[" + chinese_character + "]{3,}", chinese_character * 3, string)

    return string


if __name__ == "__main__":
    print(stemming("IGGGGGG"))
    print(stemming("66666666"))
    print(stemming("爬爬爬爬爬爬"))
```

运行结果

```
IGGG
66666666
爬爬爬
```

> 学习参考文献：知乎·优达学城回答的“自然语言处理时，通常的文本清理流程是什么？”；《自然语言处理入门》(何晗)：2.10.2