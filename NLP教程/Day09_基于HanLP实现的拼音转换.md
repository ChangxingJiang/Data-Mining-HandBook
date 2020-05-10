# Day09 : 基于HanLP实现的拼音转换

> **作者**：长行
>
> **时间**：2020.05.10

> 这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

拼音转换是指将汉字转化为拼音的过程。

因为拼音转换涉及多音字的问题，所以仍然不能仅通过字转换，还需要按词转换。

下面我们使用HanLP的方法实现拼音转换。

```python
from pyhanlp import *

if __name__ == "__main__":
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")
    text = "重要的事情重复三遍"

    pinyin_list = HanLP.convertToPinyinList(text)
    print("原文:", text)
    print("拼音(数字音调):", pinyin_list)
    print("拼音(符号音调):", [pinyin.getPinyinWithToneMark() for pinyin in pinyin_list])
    print("拼音(无音调):", [pinyin.getPinyinWithoutTone() for pinyin in pinyin_list])
    print("声调:", [pinyin.getTone() for pinyin in pinyin_list])
```

运行结果

```
原文: 重要的事情重复三遍
拼音(数字音调): [zhong4, yao4, de5, shi4, qing2, chong2, fu4, san1, bian4]
拼音(符号音调): ['zhòng', 'yào', 'de', 'shì', 'qíng', 'chóng', 'fù', 'sān', 'biàn']
拼音(无音调): ['zhong', 'yao', 'de', 'shi', 'qing', 'chong', 'fu', 'san', 'bian']
声调: [4, 4, 5, 4, 2, 2, 4, 1, 4]
```

HanLP的实现方法首先是将文本分词，然后依据从词语到拼音的词典，以词为单位将文本转换为拼音。

因此，在转换过程中，优先按词语进行转换，在没有匹配到更长的词语的情况下， 多音字默认取第一个拼音。

>  学习参考文献：《自然语言处理入门》(何晗)：2.10.3