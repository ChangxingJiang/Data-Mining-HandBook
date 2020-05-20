# Day07 : 基于HanLP实现的停用词过滤

> **作者**：长行
>
> **时间**：2020.05.10

> 这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

停用词，是指文本中没有多少实际意义的词语，包括助词、连词、副词、语气词等词性，句子中去掉了停用词并不影响语句的理解。

停用词视具体情况在英雄联盟的游戏内聊天框中，敏感词和低俗词也都被视作了停用词。

停用词过滤是语言文本预处理的一个重要步骤，有两种常用的情景：

- 将分词结果中的停用词剔除
- 直接将文本中的停用词替换为*或移除

下面，我们使用HanLP提供的双数组字典树来实现这个功能。

首先，我们从HanLP实现停用词典的加载。

```python
def load_from_file(path):
    """
    从词典文件加载DoubleArrayTrie
    :param path: 词典路径
    :return: 双数组trie树
    """
    map = JClass('java.util.TreeMap')()  # 创建TreeMap实例
    with open(path) as src:
        for word in src:
            word = word.strip()  # 去掉Python读入的\n
            map[word] = word
    return JClass('com.hankcs.hanlp.collection.trie.DoubleArrayTrie')(map)
```

停用词词典的格式为一个词一行，每行之间用换行符\n分隔。

## 提出分词结果中的停用词

针对分词结果，遍历每个词语，若它存在于停用词字典树中，则删除该词。这里使用列表生成式实现。

```python
def remove_stopwords_termlist(termlist, trie):
    return [term.word for term in termlist if not trie.containsKey(term.word)]
    
if __name__ == '__main__':
    HanLP.Config.ShowTermNature = False
    trie = load_from_file(HanLP.Config.CoreStopWordDictionaryPath)
    text = "停用词的意义相对而言无关紧要的词吧"
    segment = DoubleArrayTrieSegment()
    termlist = segment.seg(text)
    print("分词结果:", termlist)
    print("分词结果去除停用词:", remove_stopwords_termlist(termlist, trie))
```

运行结果

```
分词结果: [停用, 词, 的, 意义, 相对而言, 无关紧要, 的, 词, 吧]
分词结果去除停用词: ['停用', '词', '意义', '无关紧要', '词']
```

## 直接处理文本中的停用词

在过滤敏感词时，需要将敏感词替换为特定的字符串，例如**，这时使用不分词，直接替换的方法效率更高。这里使用HanLP的双数组字典树的getLongestSearcher方法实现。

```python
def replace_stropwords_text(text, replacement, trie):
    searcher = trie.getLongestSearcher(JString(text), 0)
    offset = 0
    result = ''
    while searcher.next():
        begin = searcher.begin
        end = begin + searcher.length
        if begin > offset:
            result += text[offset: begin]
        result += replacement
        offset = end
    if offset < len(text):
        result += text[offset:]
    return result
    
if __name__ == '__main__':
    text = "停用词的意义相对而言无关紧要的词吧"
    trie = load_from_words("的", "相对而言", "吧")
    print("不分词去掉停用词:", replace_stropwords_text(text, "**", trie))
```

运行结果

```
不分词去掉停用词: 停用词\*\*意义\*\*无关紧要\*\*词\*\*
```

>  学习使用教材：《自然语言处理入门》(何晗)：2.10