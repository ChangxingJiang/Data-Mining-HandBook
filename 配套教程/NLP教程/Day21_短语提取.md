# Day21 : 短语提取

> **作者**：长行
>
> **时间**：2020.05.10

将基于信息熵和互信息的新词提取方法（第20天）中的字符替换为单词，即可将其转换为短语识别的方法。

下面我们仍然使用神超直播间的弹幕的8个小时的时间切片作为例子，使用HanLP中的extractPhrase方法实现。

```python
from pyhanlp import *
from utils import file

def extract(corpus):
    text = file.as_string(corpus)  # 工具类:将文件读取为str
    phrase_info_list = HanLP.extractPhrase(text, 20)

    for phrase in phrase_info_list:
        print(phrase, end=",")

if __name__ == "__main__":
    extract("data/神超直播间弹幕切片.txt")
```

> （HanLP的短语提取模块仅支持二元语法短语，暂不支持n元语法短语的提取）

**运行结果**

```
超哥,神超,鬼书,吃鸡,复活甲,狂战,影剑圣,轮子妈,雷霆劫,皎月,剑转,掠食者,正义手,法转,发牌员,神装,星炼金,玩影,婕拉,玩游侠,
```

可以看到，其中有一些“鬼书”、“狂战”等词语其实不能算作是词语，这是因为我没把英雄联盟领域的领域词典导入给分词器，导致对分词器来说，“鬼”、“书”、“狂”、“战”等字都是单字成词，所以将其识别为了短语。

因此，对新领域的短语提取应该在领域词典的基础上操作。

> 学习参考文献：《自然语言处理入门》(何晗)：9.3