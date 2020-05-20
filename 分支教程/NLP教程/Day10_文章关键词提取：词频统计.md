# Day10 : 文章关键词提取：词频统计

> **作者**：长行
>
> **时间**：2020.05.10

关键词提取是词语颗粒度的信息抽取的一种重要的需求，即提取文章中重要的词语。

关键词提取的常用方法包括词频统计、TF-IDF和TextRank等。

其中，词频和TextRank属于单文档算法，即只需一篇文章即可提取出其中的关键词；而TF-IDF则属于多文档宣发，需要其他文档的辅助来提取当前文章的关键词。

## 词频统计的Python实现

词频统计的逻辑是：在一篇文章中，越重要的关键词往往会在文章中反复出现；因为为了解释关键词，作者经常会反复地提及它们。所以通过统计文章中各个词语的出现频率，即可初步地获得关键词。

但是因为齐夫定律，文章中出现频率最高的往往并不是长度较长的关键词，而是标点符号和助词等，因此在词频统计之前还需要先进行停用词过滤。

> 齐夫定律：一个单词的词频与它的词频排名成反比。

由此，词频统计的流程通常是中文分词、停用词过滤、词频统计。依据以上逻辑，我在Python中实现以下词频统计。（以《红楼梦·桃花行》节选为例）

```python
from pyhanlp import HanLP
from pyhanlp import JClass

def load_from_words(*words):
    """
    从词汇构造双数组trie树
    :param words: 一系列词语
    :return:
    """
    map = JClass('java.util.TreeMap')()  # 创建TreeMap实例
    for word in words:
        map[word] = word
    return JClass('com.hankcs.hanlp.collection.trie.DoubleArrayTrie')(map)

def remove_stopwords_termlist(termlist, trie):
    return [term.word for term in termlist if not trie.containsKey(term.word)]

if __name__ == "__main__":
    # 《红楼梦·桃花行》节选
    article = "桃花帘外东风软，桃花帘内晨妆懒。帘外桃花帘内人，人与桃花隔不远。"
    # 停用词表(诗中包含的哈工大停用词表的停用词)
    trie = load_from_words("，", "。", "与")

    # 中文分词+停用词过滤
    termlist = HanLP.segment(article)
    termlist = remove_stopwords_termlist(termlist, trie)  # 分词结果去除停用词
    print("分词结果:", termlist)

    # 词频统计
    word_frequency = dict()
    for word in termlist:
        if word not in word_frequency:
            word_frequency[word] = 0
        word_frequency[word] += 1
    word_frequency_sorted = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)  # 词频排序
    for i in range(5):
        print(word_frequency_sorted[i][0], "词频:", word_frequency_sorted[i][1])
```

> 其中load_from_words和remove_stopwords_termlist在之前（第7天）的学习中已经掌握。

**运行结果**

```
分词结果: ['桃花', '帘', '外', '东风', '软', '桃花', '帘', '内', '晨妆', '懒', '帘', '外', '桃花', '帘', '内', '人', '人', '桃花', '隔', '不远']
桃花 词频: 4
帘 词频: 4
外 词频: 2
内 词频: 2
人 词频: 2
```

**基于HanLP实现的词频统计**

HanLP中封装了TermFrequencyCounter类用来统计文档的词频，接着我们使用这个类实现词频统计。

```python
from pyhanlp import *

TermFrequency = JClass('com.hankcs.hanlp.corpus.occurrence.TermFrequency')
TermFrequencyCounter = JClass('com.hankcs.hanlp.mining.word.TermFrequencyCounter')

if __name__ == '__main__':
    counter = TermFrequencyCounter()
    counter.add("桃花帘外东风软，桃花帘内晨妆懒。帘外桃花帘内人，人与桃花隔不远。")  # 第1个文档
    counter.add("东风有意揭帘栊，花欲窥人帘不卷。桃花帘外开仍旧，帘中人比桃花瘦。")  # 第2个文档
    print("2篇文章的词频前5名:", counter.top(5))

    #  根据词频提取关键词
    print("第1篇文章的词频前5名:", TermFrequencyCounter.getKeywordList("桃花帘外东风软，桃花帘内晨妆懒。帘外桃花帘内人，人与桃花隔不远。", 5))
```

运行结果

```
2篇文章的词频前5名: [帘=8, 桃花=6, 外=3, 东风=2, 隔=1]
第1篇文章的词频前5名: [桃花, 帘, 外, 隔, 软]
```

可以看到，整体结果是相近的，HanLP去除了更多的停用词，包括“人”、“内”以及标点符号等。

用词频提取关键词存在一个缺陷，就是即使使用过滤停用词以后，高频词也并与关键词完全等价。例如在分析一个明星的相关新闻时，明星名字的出现频率可能是最高的，但是在我们希望找到每一篇文章各自的特点，而不是文章的共性，此时，我们就需要引入TF-IDF等关键词提取方法。

> 学习参考文献：《自然语言处理入门》(何晗)：9.2.1