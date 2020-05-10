# Day03 : 字典树

> **作者**：长行
>
> **时间**：2020.05.08

> 通过《自然语言处理入门》(何晗)的第2章来学习一下分词的常用算法，因此以下的实现方法都是通过HanLP实现的。这里主要记录我在学习过程中整理的知识、调试的代码和心得理解，以供其他学习的朋友参考。

字符串集合常用字典树（Trie树）存储，这是一种字符串上的树形数据结构。

字典树中每条边都对应一个字，从根节点往下的路径构成一个个字符串。字典树并不直接在节点上存储字符串，而是将词典视作根节点到某节点之间的一条路径，并在终点节点上做个“该节点对应词语的结尾”的标记。

字符串是一条路径，要查询一个单词，只需顺着这条路径从归根节点往下走。如果能走到特殊标记的节点，则说明该字符串在集合中，否则说明不存在。

## 字典树的节点实现
每个节点都应该知道自己的子节点与对应的边，以及自己是否对应一个词。
我们用None来表示该节点不对应词语。

节点的Python实现如下：

```python
class Node(object):
    def __init__(self, value):
        self._children = {}  # 子节点存储变量
        self._value = value  # 当前节点的值

    def _add_child(self, char, value, overwrite=False):
        child = self._children.get(char)
        if child is None:  # 判断当前节点是否已经存在字符char对应的child
            child = Node(value)
            self._children[char] = child
        elif overwrite:  # 根据overwrite判断是否覆盖child的值
            child._value = value
        return child
```

## 字典树的增删改查实现

只要将以上节点连到根节点上，就得到了字典树。根节点继承自普通节点，并增加了一些面向用户的公开方法。因此，只要拿到根节点，就能抓住整棵字典树。

从确定有限状态自动机（DFA）的角度来讲，每个节点都是一个状态，状态表示当前已查询到的前缀。

> **有限状态自动机（DFA）**\
> 有限状态自动机（deterministic finite automaton, DFA）是一个能实现状态转移的自动机。对于一个给定的属于该自动机的状态和一个属于该自动机字母表Σ的字符，它都能根据事先给定的转移函数转移到下一个状态（这个状态可以是先前那个状态）。每次输入都会引起状态的改变或者不变。\
>参考文献：https://www.cnblogs.com/dh-dh/p/10245474.html

字典树的Python实现如下：

```python
class Trie(Node):
    def __init__(self):
        super().__init__(None)

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self
        for char in key:  # 遍历字符串中的每一个字符
            state = state._children.get(char)  # 一直依据路径找到目标词
            if state is None:
                return None
        return state._value

    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key):  # 枚举字符串中的每一个字符及其位置坐标
            if i < len(key) - 1:  # 若当前词不是结尾词
                state = state._add_child(char, None, False)
            else:  # 若当前词是结尾词
                state = state._add_child(char, value, True)
```

## 运行测试

我们对以上的字典树做如下测试：

```python
if __name__ == '__main__':
    trie = Trie()
    # 增
    trie['自然'] = 'nature'
    trie['自然人'] = 'human'
    trie['自然语言'] = 'language'
    trie['自语'] = 'talk to oneself'
    trie['入门'] = 'introduction'
    print(trie['自然'])
    # 删
    trie['自然'] = None
    print(trie['自然'])
    # 改
    print(trie['自然语言'])
    trie['自然语言'] = 'human language'
    print(trie['自然语言'])
```

运行结果

```
nature
None
language
human language
```

字典树的使用是为了提高字典的查询速度，优化分词算法的效率。

>学习使用教材：《自然语言处理入门》(何晗)：2.4.1 - 2.4.3\
>本文中代码大部分引自该书中的代码