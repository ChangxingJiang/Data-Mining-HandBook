# Day02 : 中文词典分词的切分算法

> **作者**：长行
>
> **时间**：2020.05.08

> 我以前用过Jieba、Pkuseg、HanLP等开源工具的分词功能，现在主要通过《自然语言处理入门》(何晗)的第2章来学习一下分词的常用算法，因此以下的实现方法都是通过HanLP实现的。这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

**中文分词**指的是将一段文本拆分为一系列单词的过程，将这些单词顺序拼接后等于原文本。

中文分词算法大致分为基于词典规则和基于机器学习这两大派别。**词典分词**是最简单、最常见的分词算法，仅需一部词典和一套查词典的规则即可。

下面主要介绍以下4种切分算法：完全切分、正向最长匹配、逆向最长匹配、双向最长匹配。

## 加载字典

我们载入HanLP的核心词典的词语列表作为研究对象。

```python
from pyhanlp import HanLP
from pyhanlp import JClass

def load_dictionary():
    """
    加载HanLP中的mini词库
    :return: 一个set形式的词库
    """
    IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')  # ①
    path = HanLP.Config.CoreDictionaryPath.replace('.txt', '.mini.txt')  # ②
    dic = IOUtil.loadDictionary([path])  # ③
    return set(dic.keySet())

if __name__ == "__main__":
    dic = load_dictionary()
    print("词典大小:", len(dic))
    print("词典中的第1个词:", list(dic)[0])
```
1. 利用HanLP提供的JClass取得HanLP中的IOUtil工具类
2. 获取HanLP的设置项Config中的词典路径
3. 调用IOUtil的静态方法loadDictionary；其返回值为一个字典对象（Java的Map对象）；这个字典的key为词语，value为词语的词性和频数；因为这里我们只需要词语列表，所以使用keySet获取Set对象的词语列表。

**运行结果**

```
词典大小: 85584 词典中的第1个词: 分流
```

## 完全切分

**完全切分**指的是，找出一段文本中所有单词，无论这个词在这个句子中是否是一个词。

朴素的完全切分算法的实现逻辑，是遍历文本中所有的连续序列，并查询该序列是否存在于词典中。

```python
import load_dictionary # 载入加载词典的函数

def fully_segment(text, dic):
    word_list = []
    for i in range(len(text)):  # 遍历text中的所有位置下标
        for j in range(i + 1, len(text) + 1):  # 遍历[i + 1, len(text)]区间
            word = text[i:j]  # 取出连续区间[i, j]对应的字符串
            if word in dic:  # 如果存在于词典中，则认为是一个词
                word_list.append(word)
    return word_list

if __name__ == '__main__':
    dic = load_dictionary()
    print(fully_segment('项目的研究', dic))
    print(fully_segment('商品和服务', dic))
    print(fully_segment('研究生命起源', dic))
    print(fully_segment('当下雨天地面积水', dic))
    print(fully_segment('结婚的和尚未结婚的', dic))
    print(fully_segment('欢迎新老师生前来就餐', dic))
```

**运行结果**

```
['项', '项目', '目', '目的', '的', '研', '研究', '究']
['商', '商品', '品', '和', '和服', '服', '服务', '务']
['研', '研究', '研究生', '究', '生', '生命', '命', '起', '起源', '源']
['当', '当下', '下', '下雨', '下雨天', '雨', '雨天', '天', '天地', '地', '地面', '面', '面积', '积', '积水', '水']
['结', '结婚', '婚', '的', '和', '和尚', '尚', '尚未', '未', '结', '结婚', '婚', '的']
['欢', '欢迎', '迎', '迎新', '新', '老', '老师', '师', '师生', '生', '生前', '前', '前来', '来', '就', '就餐', '餐']
```

在实际研究中，我们需要的并不是“完全切分”中得到文本中所有出现在词典中的单词所构成的链表，而是一个个确切的词组组成的句子。例如，我们希望“北京大学”是一个完整的词，而非“北京+大学”两个碎片。

为此，我们需要完善一下我们的规则，考虑到越长的单词表达的意义越丰富，于是我们定义单词越长**优先级**越高。

具体来说，就是在某个下标为起点递增查词的过程中，优先输出更长的单词，这种规则被称为最长匹配算法。该下标的扫描顺序如果是从前往后，则称正向最长匹配，反之则称逆向最长匹配。

## 正向最长匹配

```python
import load_dictionary # 载入加载词典的函数

def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):  # i 当前扫描位置在text中的位置下标
        longest_word = text[i]  # 当前扫描位置的单字
        for j in range(i + 1, len(text) + 1):  # j 遍历当前扫描位置中所有可能的结尾
            word = text[i:j]  # 从当前位置到结尾的连续字符串
            if word in dic:  # 如果存在于词典中
                if len(word) > len(longest_word):  # 词语越长优先级越高
                    longest_word = word
        word_list.append(longest_word)  # 输出最长词
        i += len(longest_word)  # 正向扫描:将词语长度加到扫描位置的位置下标中
    return word_list

if __name__ == '__main__':
    dic = load_dictionary()
    print(forward_segment('项目的研究', dic))
    print(forward_segment('商品和服务', dic))
    print(forward_segment('研究生命起源', dic))
    print(forward_segment('当下雨天地面积水', dic))
    print(forward_segment('结婚的和尚未结婚的', dic))
    print(forward_segment('欢迎新老师生前来就餐', dic))
```

**运行结果**

```
['项目', '的', '研究']
['商品', '和服', '务']
['研究生', '命', '起源']
['当下', '雨天', '地面', '积水']
['结婚', '的', '和尚', '未', '结婚', '的']
['欢迎', '新', '老师', '生前', '来', '就餐']
```

## 逆向最长匹配

因为正向最长匹配很容易遇到诸如“研究生命起源”中因“研究生”优先级大于“研究”的误差，因此，有人提出了逆向匹配的方法。

```python
import load_dictionary  # 载入加载词典的函数

def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1  # 从text中的最后一个字开始扫描
    while i >= 0:  # i 当前扫描位置在text中的位置下标
        longest_word = text[i]  # 扫描位置的单字
        for j in range(0, i):  # 遍历[0, i]区间中所有可能的起点
            word = text[j: i + 1]  # 取出[j, i]区间作为待查询单词
            if word in dic:  # 如果存在于词典中
                if len(word) > len(longest_word):  # 词语越长优先级越高
                    longest_word = word
                    break
        word_list.insert(0, longest_word)  # 逆向扫描，所以越先查出的单词在位置上越靠后
        i -= len(longest_word)  # 正向扫描:将词语长度减到扫描位置的位置下标中
    return word_list

if __name__ == '__main__':
    dic = load_dictionary()
    print(backward_segment('项目的研究', dic))
    print(backward_segment('商品和服务', dic))
    print(backward_segment('研究生命起源', dic))
    print(backward_segment('当下雨天地面积水', dic))
    print(backward_segment('结婚的和尚未结婚的', dic))
    print(backward_segment('欢迎新老师生前来就餐', dic))
```

**运行结果**

```
['项', '目的', '研究']
['商品', '和', '服务']
['研究', '生命', '起源']
['当', '下雨天', '地面', '积水']
['结婚', '的', '和', '尚未', '结婚', '的']
['欢', '迎新', '老', '师生', '前来', '就餐']
```

## 双向最长匹配

逆向最长匹配虽然拆分“研究生命起源”得到了正确结果，但是在拆分“项目的研究”时却出现了新的问题。因此，有的人又提出了综合两种规则，期待它们取长补短的方法，称之为双向匹配。

双向匹配的逻辑流程如下：

1. 同时执行正向和逆向最长匹配
2. 若两者的词数不同，则返回次数更少的哪一个
3. 若两者的词数相同，则返回两者中单字更少的哪一个
4. 若两者的单字数也相同，则优先返回逆向最长匹配的结果

这种规则的出发点来自语言学中的启发——汉字中单字词的数量要远远小于非单字词。因此，算法应当尽量减少结果中的单字，保留更多的完整词语，这样的算法也称**启发式算法**。

```python
from book.N001_HanLP_Load_dictionary import load_dictionary  # 载入加载词典的函数
from book.N003_HanLP_Forward_Segment import forward_segment  # 载入正向最长匹配的函数
from book.N004_HanLP_Backward_segment import backward_segment  # 载入逆向最长匹配的函数

def count_single_char(word_list: list):  # 统计单字成词的个数
    return sum(1 for word in word_list if len(word) == 1)

def bidirectional_segment(text, dic):
    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    if len(f) < len(b):  # 词数更少优先级更高
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):  # 单字更少优先级更高
            return f
        else:
            return b  # 都相等时逆向匹配优先级更高

if __name__ == '__main__':
    dic = load_dictionary()
    print(bidirectional_segment('项目的研究', dic))
    print(bidirectional_segment('商品和服务', dic))
    print(bidirectional_segment('研究生命起源', dic))
    print(bidirectional_segment('当下雨天地面积水', dic))
    print(bidirectional_segment('结婚的和尚未结婚的', dic))
    print(bidirectional_segment('欢迎新老师生前来就餐', dic))
```

**运行结果**

```
['项', '目的', '研究']
['商品', '和', '服务']
['研究', '生命', '起源']
['当下', '雨天', '地面', '积水']
['结婚', '的', '和', '尚未', '结婚', '的']
['欢', '迎新', '老', '师生', '前来', '就餐']
```

总体来说，这三种匹配方法都不能正确地解决所有的例子，由此可以看到基于词典规则的系统的脆弱。

根据《自然语言处理入门》中的评测，这三种匹配方式的运行速度大致如下：

* 正向匹配和逆向匹配的速度差不多，是双向匹配速度的两倍
* Python的运行速度比Java慢，效率只有Java的一半不到

> 学习使用教材：《自然语言处理入门》(何晗)：2.1 - 2.3 \
> 本文中代码大部分引自该书中的代码，个人还是很推荐这本书的，确实是非常好的教材。