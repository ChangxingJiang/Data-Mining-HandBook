# Day22 : 英文词形还原（各处理库准确率评测）

> **作者**：长行
>
> **时间**：2020.05.10

“词形还原”(lemmatization)是指去掉单词的词形词缀，返回单词原型。

例如对于动词来说，就是去掉动词的过去式、过去完成时等形式，返回动词原型；对于名词来说，则是去掉名词的复数形式，返回名词原型；对于形容词来说，则是去掉比较级、最高级等形式，返回形容词原型。

“词性还原”与“词干提取”(stemming)的区别在于：词干提取的结果可能不是完整的词，而词性还原的结果则是具有一定意义的、完整的词语。

下面我们对NLTK、spaCy和LemmInflect的准确率进行评测。我们使用Automatically Generated Inflection Database (AGID)作为基准来进行评测。

> Automatically Generated Inflection Database的下载地址：http://wordlist.aspell.net/other/

准确率评测结果：

```
| Package          | Accuracy |
|-----------------------------|
| LemmInflect      |  94.66%  |
| NLTK             |  83.72%  |
| spaCy            |  65.98%  |
|-----------------------------|
```

## 基于LemmInflect的词形还原实现

> Github地址：https://github.com/bjascob/LemmInflect

```python
def by_lemminflect(word, pos):
    """ 使用lemminflect.getLemma实现词性还原
    :param word: <str> 单词
    :param pos: <str> 词性
    :return: <str> 词性还原结果
    """
    if pos.lower() == "a":
        pos = "ADJ"
    elif pos.lower() == "n":
        pos = "NOUN"
    elif pos.lower() == "v":
        pos = "VERB"
    elif pos.lower() == "p":
        pos = "PROPN"
    return getLemma(word, pos)

if __name__ == "__main__":
    test_word_list = [["men", "n"], ["computer", "n"], ["ate", "v"], ["running", "v"], ["saddest", "a"],
                      ["fancier", "a"]]
    for test_word in test_word_list:
        print(by_lemminflect(test_word[0], test_word[1]))
```

运行结果

```
('man',)
('computer',)
('eat',)
('run',)
('sad',)
('fancy',)
```

## 基于NLTK的词形还原实现

NLTK在Github上拥有8800星标，其中stem.WordNetLemmatizer用于实现词形还原。

> Github地址：https://github.com/nltk/nltk

```python
def by_nltk(word, pos):
    """ 使用nltk.stem.WordNetLemmatizer实现词性还原
    :param word: <str> 单词
    :param pos: <str> 词性
    :return: <str> 词性还原结果
    """
    return nltk_wnl.lemmatize(word, pos.lower())

if __name__ == "__main__":
    test_word_list = [["men", "n"], ["computer", "n"], ["ate", "v"], ["running", "v"], ["saddest", "a"],
                      ["fancier", "a"]]
    for test_word in test_word_list:
        print(by_nltk(test_word[0], test_word[1]))
```

运行结果

```
men
computer
eat
run
sad
fancy
```

