# Day22 : spaCy实现的英文词形还原

> **作者**：长行
>
> **时间**：2020.05.10

“词形还原”(lemmatization)是指去掉单词的词形词缀，返回单词原型。

spaCy是在Github上16000星标的自然语言处理库，使用lemma_属性支持词形还原。

> spaCy是基于Python和Cython中自然语言处理库。spaCy包含预训练的统计模型、词向量等，支持50多种语言。
>
> Github地址：https://github.com/explosion/spaCy

## SpaCy的安装方法

spaCy需要在pip安装之后，再下载对应的语言模型才可以正常运行。以英文核心语料库为例，spaCy的安装命令如下：

```
pip install spacy
python -m spacy download en_core_web_sm
```

## SpaCy的词形还原实现

```python
def by_spacy(word):
    """ 使用spacy实现词性还原
    :param word: <str> 单词
    :return: <str> 词性还原结果
    """
    return spacy_tool(word)[0].lemma_

if __name__ == "__main__":
    test_word_list = [["men", "n"], ["computer", "n"], ["ate", "v"], ["running", "v"], ["saddest", "a"], ["fancier", "a"]]
    for test_word in test_word_list:
        print(by_spacy(test_word[0]))
```

运行结果

```
man
computer
eat
run
saddest
fancy
```

可以看到在测试词语的词形还原无误。

**学习NLP的第22天——英文词形还原（各处理库准确率评测）**

**副标题：英文词形还原的Python实现（各处理库准确率评测）**

“词形还原”(lemmatization)是指去掉单词的词形词缀，返回单词原型。

例如对于动词来说，就是去掉动词的过去式、过去完成时等形式，返回动词原型；对于名词来说，则是去掉名词的复数形式，返回名词原型；对于形容词来说，则是去掉比较级、最高级等形式，返回形容词原型。

“词性还原”与“词干提取”(stemming)的区别在于：词干提取的结果可能不是完整的词，而词性还原的结果则是具有一定意义的、完整的词语。

**—————————————————————————————————————————**

下面我们对NLTK、spaCy和LemmInflect的准确率进行评测。我们使用Automatically Generated Inflection Database (AGID)作为基准来进行评测。

Automatically Generated Inflection Database的下载地址：http://wordlist.aspell.net/other/

准确率评测结果：

| Package          | Accuracy | |-----------------------------| | LemmInflect      |  94.66%  | | NLTK             |  83.72%  | | spaCy            |  65.98%  | |-----------------------------|

**基于LemmInflect的词形还原实现**

Github地址：https://github.com/bjascob/LemmInflect

def by_lemminflect(word, pos):    """ 使用lemminflect.getLemma实现词性还原    :param word: <str> 单词    :param pos: <str> 词性    :return: <str> 词性还原结果    """    if pos.lower() == "a":        pos = "ADJ"    elif pos.lower() == "n":        pos = "NOUN"    elif pos.lower() == "v":        pos = "VERB"    elif pos.lower() == "p":        pos = "PROPN"    return getLemma(word, pos) if __name__ == "__main__":    test_word_list = [["men", "n"], ["computer", "n"], ["ate", "v"], ["running", "v"], ["saddest", "a"],                      ["fancier", "a"]]    for test_word in test_word_list:        print(by_lemminflect(test_word[0], test_word[1]))

运行结果

('man',) ('computer',) ('eat',) ('run',) ('sad',) ('fancy',)

**基于NLTK的词形还原实现**

NLTK在Github上拥有8800星标，其中stem.WordNetLemmatizer用于实现词形还原。

Github地址：https://github.com/nltk/nltk

def by_nltk(word, pos):    """ 使用nltk.stem.WordNetLemmatizer实现词性还原    :param word: <str> 单词    :param pos: <str> 词性    :return: <str> 词性还原结果    """    return nltk_wnl.lemmatize(word, pos.lower()) if __name__ == "__main__":    test_word_list = [["men", "n"], ["computer", "n"], ["ate", "v"], ["running", "v"], ["saddest", "a"],                      ["fancier", "a"]]    for test_word in test_word_list:        print(by_nltk(test_word[0], test_word[1]))

运行结果

men computer eat run sad fancy

**基于spaCy的词形还原实现**

spaCy的安装方法详见“学习NLP的第22天——spaCy实现的英文词形还原”