# Day14 : 语言模型的训练与预测

> **作者**：长行
>
> **时间**：2020.05.08

简单来说，语言模型就是通过统计已经分词的语料库中一元语法和二元语法出现的频次，实现对句子的分词。

下面我们使用HanLP的相关模块实现语言模型的训练和预测。

## 训练

训练是指根据给定样本集估计模型参数的过程，简单来说，在语言模型中就是统计二元语法出现的频次和一元语法出现的频次。

首先，我们使用HanLP的CorpusLoader.convert2SentenceList加载语料库（空格分词格式）。

```python
from pyhanlp import *
CorpusLoader = SafeJClass("com.hankcs.hanlp.corpus.document.CorpusLoader")  # 语料库加载Java模块

corpus_path = "data\my_cws_corpus.txt"  # 语料库所在路径
sentences = CorpusLoader.convert2SentenceList(corpus_path) # 返回List<List<IWord>>类型
for sent in sentences:
    print(sent)
```

运行结果：
```
[商品, 和, 服务]
[商品, 和服, 物美价廉]
[服务, 和, 货币]
```

接着，我们使用HanLP的NatureDictionaryMaker统计一元语法和二元语法。

```python
NatureDictionaryMaker = SafeJClass("com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker")  # 词典模型Java模块(统计一元、二元语法)

model_path = "data\my_cws_model"  # 语言模型存储路径
for sent in sentences:
    for word in sent:
        if word.label is None:
            word.setLabel("n")  # 赋予每个单词一个虚拟的n词性用作占位
maker = NatureDictionaryMaker()  # 构造NatureDictionaryMaker对象
maker.compute(sentences)  # 统计句子中的一元语法、二元语法
maker.saveTxtTo(model_path)  # 将统计结果存储到路径
```

运行后在程序目录中的data目录中，新建一元语法模型（my_cws_model.txt）、二元语法模型（my_cws_model.ngram.txt）和词性标注相关文件（my_cws_model.tr.txt）文件。

## 预测
预测是指利用模型对样本进行推断的过程，简单来说，就是通过我们之前统计的一元语法和二元语法的频次，推断句子的分词序列。

首先，我们使用HanLP的CoreDictionary和CoreBiGramTableDictionary加载刚才训练的语言模型。

```python
HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # 一元语法模型路径
HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # 二元语法模型路径
CoreDictionary = LazyLoadingJClass("com.hankcs.hanlp.dictionary.CoreDictionary")  # 加载一元语法模型Java模块
CoreBiGramTableDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary")  # 加载二元语法模型Java模块
print(CoreDictionary.getTermFrequency("商品"))  # 测试"商品"的一元语法频次
print(CoreBiGramTableDictionary.getBiFrequency("商品", "和"))  # 测试"商品 和"的二元语法频次
```

运行结果：

```
2
1
```

词网是HanLP提出的特指句子中所有一元语法构成的网状结构的概念，其生成过程为：根据一元语法词典，将句子中所有单词找出来，并将起始位置相同的单词写作一行。例如，“商品和服务“的词网如下：

```
0:[ ]
1:[商品]
2:[]
3:[和, 和服]
4:[服务]
5:[务]
6:[ ]
```

下面，我们使用HanLP的WordNet模块构建词网。
```python
from jpype import JString
WordNet = JClass("com.hankcs.hanlp.seg.common.WordNet")  # 构建词网的Java模块(词网模块)
Vertex = JClass("com.hankcs.hanlp.seg.common.Vertex")  # 构建词网的Java模块(词语存储对象)

sent = "货币和服务"
trie = CoreDictionary.trie

# 生成词网
searcher = trie.getSearcher(JString(sent), 0)
wordnet = WordNet(sent)
while searcher.next():
    wordnet.add(searcher.begin + 1,
                Vertex(sent[searcher.begin:searcher.begin + searcher.length], searcher.value, searcher.index))

# 补充一元语法中不包含但是保证图联通必须的词
vertexes = wordnet.getVertexes()
i = 0
while i < len(vertexes):
    if len(vertexes[i]) == 0:  # 空白行
        j = i + 1
        for j in range(i + 1, len(vertexes) - 1):  # 寻找第一个非空行 j
            if len(vertexes[j]):
                break
        wordnet.add(i, Vertex.newPunctuationInstance(sent[i - 1: j - 1]))  # 填充[i, j)之间的空白行
        i = j
    else:
        i += len(vertexes[i][-1].realWord)

print(wordnet)
```

运行结果

```
0:[ ]
1:[货币]
2:[]
3:[和, 和服]
4:[服务]
5:[务]
6:[ ]
```

最后，我们使用维特比算法来计算词网中的最短路径，以“货币和服务”的句子为例，得出其分词序列预测结果。

```python
def viterbi(wordnet):
    nodes = wordnet.getVertexes()
    # 前向遍历
    for i in range(0, len(nodes) - 1):
        for node in nodes[i]:
            for to in nodes[i + len(node.realWord)]:
                to.updateFrom(node)  # 根据距离公式计算节点距离，并维护最短路径上的前驱指针from
    # 后向回溯
    path = []  # 最短路径
    f = nodes[len(nodes) - 1].getFirst()  # 从终点回溯
    while f:
        path.insert(0, f)
        f = f.getFrom()  # 按前驱指针from回溯
    return [v.realWord for v in path]

print(viterbi(wordnet))
```

运行结果

```
[' ', '货币', '和', '服务', ' ']
```

>学习参考文献：《自然语言处理入门》(何晗)：3.3-3.4.4







