# Day04 : HanLP词典分词的Python实现

> **作者**：长行
>
> **时间**：2020.05.10

>通过《自然语言处理入门》(何晗)的第2章来学习HanLP词典分词的Python实现。这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

首先，我们导入HanLP

```python
from pyhanlp import HanLP
```

下面，实现最基本的中文分词（依据HanLP默认词典，显示词语和词性、处理数字和英文的识别）

```python
sentence = "deadline并不能帮你提升quality，只是给了你交一大堆trash上去的勇气"
print(HanLP.segment(sentence))
```
运行结果
```
[deadline/nx, 并/cc, 不能/v, 帮/v, 你/rr, 提升/v, quality/nx, ，/w, 只是/d, 给/p, 了/ule, 你/rr, 交/v, 一大/n, 堆/v, trash/nx, 上去/vf, 的/ude1, 勇气/n]
```

下面我们来尝试实现一些更多设置

## 关闭词性显示
```python
HanLP.Config.ShowTermNature = False
sentence = "deadline并不能帮你提升quality，只是给了你交一大堆trash上去的勇气"
print(HanLP.segment(sentence))
```

运行结果

```
[deadline, 并, 不能, 帮, 你, 提升, quality, ，, 只是, 给, 了, 你, 交, 一大, 堆, trash, 上去, 的, 勇气]
```

## 分别获取分词结果中的词语与词性
```python
sentence = "deadline并不能帮你提升quality，只是给了你交一大堆trash上去的勇气"
for term in HanLP.segment(sentence):
    print("单词:%s 词性:%s" % (term.word, term.nature))
```

运行结果

```
单词:deadline 词性:nx
单词:并 词性:cc
......
```

## 用户词典
```python
sentence = "联合国秘书长感谢中国人民"  # 2020.02.25微博热搜榜第1名
print("设置用户词典前:", HanLP.segment(sentence))
custom_dictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
custom_dictionary.add("联合国秘书长")  # 将"联合国秘书长"设置为用户词典
print("设置用户词典后:", HanLP.segment(sentence))
```

运行结果
```
设置用户词典前: [联合国/nt, 秘书长/nnt, 感谢/v, 中国/ns, 人民/n]
设置用户词典后: [联合国秘书长/nz, 感谢/v, 中国/ns, 人民/n]
```

## 包含词性的用户词典
```python
sentence = "联合国秘书长感谢中国人民"
print("设置用户词典前:", HanLP.segment(sentence))
custom_dictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
custom_dictionary.insert("联合国秘书长", "nnt 1")  # 插入附带词性的用户词典
print("设置用户词典后:", HanLP.segment(sentence))
```

运行结果
```
设置用户词典前: [联合国/nt, 秘书长/nnt, 感谢/v, 中国/ns, 人民/n]
设置用户词典后: [联合国秘书长/nnt, 感谢/v, 中国/ns, 人民/n]
```

> 学习使用教材：《自然语言处理入门》(何晗)：2.8、《停用词与用户自定义词典》Font Tian在CSDN的博客\
> 本文中代码大部分引自该书中的代码