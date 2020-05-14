# Day-12 : 错误和调试

> **作者**：长行
>
> **时间**：2020.05.14

在我们的程序运行中，不可避免地会出现各种各样的错误。造成这些错误的原因也多种多样，可能是因为代码逻辑存在疏漏，某一种情况我们没有考虑到；也可能是因为用户的特殊操作，使我们的程序陷入了未曾涉及的场景……

## 错误处理

当Python将错误信息输出到控制台后，我们首先需要定位到错误发生的位置。

```python
def test(name):
    return int(name)
test("cbekd")
```

例如，以上代码运行时会出现如下错误信息：

```
Traceback (most recent call last):
  File "test.py", line 3, in <module>
    test()
  File "test.py", line 2, in test
    return int("cbekd")
ValueError: invalid literal for int() with base 10: 'cbekd'
```

以上错误信息就是发生这个错误之前的调用情况的跟踪信息，由上向下是从程序开始运行到发生错误的过程。在上例中：第2行-第3行是我们调用```test()```方法的过程，第4行-第5行是因为我们将非数字字符串转换为数值类型产生错误的位置，第6行为错误的提示信息。因此，我们需要解决的就是代码中第2行的错误。

面对这些各种各样的问题，在初学阶段，我们可以通过直接百度错误的提示信息，就能找到这个错误发现的原因，进而找到解决这个问题的方法。

但是当我们逐渐开始学习一些更复杂、更小众的模块时，在百度上就越来越难直接找到解决方案。这时，我们就需要开始习惯直接使用错误提示或参考文档等方式来解决问题。

## try...except...

下面我们来了解Python提供的错误处理方法（```try...except...finally...```）；当我们认为某些代码可能会出现错误，导致后续的代码无法正常运行时，我们可以使用```try```来运行这段代码；此时，如果确实发生了错误，那么```try```所包含的代码中的后续代码将不会运行，而是直接跳转至```except```中的代码；无论在运行```try```中的代码时是否出现错误，```finally```中的代码均会被执行。例如：

```python
try:
    print("try...before...")
    int("cbekd")
    print("tyy...after...")
except ValueError as e:
    print("except...", "detail:", e)
finally:
    print("finally...")
```

运行结果：

```
try...before...
except... detail: invalid literal for int() with base 10: 'cbekd'
finally...
```

每一个```try```都可以有多个不同的except，分别来处理不同类型的错误情况。Python中所有的错误都是BaseException的子类，所以当我们无法预知代码会出现什么样的错误、或者希望处理所有可能的错误时，可以使用```except BaseException```来处理所有的错误情况。

Python文档中对错误之间关系的说明详见：https://docs.python.org/3/library/exceptions.html#exception-hierarchy

## 调试技巧

有的时候，我们遇到的问题虽然找到了问题发生的位置，但是并不是问题根源的位置，例如有的时候我们发现错误是因为某个变量错误的值导致的，但是我们却不知道这个变量究竟是从哪里开始错的。此时，我们可以通过```print()```或```assert```等方法将各阶段中变量的值输出到控制台，进而寻找错误究竟是从哪一步开始发生的，这种调试方法我们通常称之为“断点调试”。

此外，Python内置的pdb模块提供了交互性的代码调试功能，包括设置断点、单步调试等，可以用来辅助我们的断点调试。例如，当在代码中添加了```pdb.set_trace()```后直接运行脚本，代码会停留在```pdb.set_trace()```的位置，等待我们输入命令。

pdb常用的命令如下：

| pdb命令 | 作用 |
| -----| -----|
| n | 执行下一行代码 |
| r | 执行代码到当前函数结束 |
| c | 执行代码到下一个断点 |
| p 变量名 | 在控制台打印变量的值 |
| l | 查看断点附近的代码 |
| b 行号 | 动态添加断点 |

例如，在之前例子中，我们使用pdb添加了断点，其代码如下。

```python
import pdb

def test(name):
    pdb.set_trace()
    return int(name)

test("cbekd")
```

运行运行以上代码，得到如下结果（其中```(Pdb)```后的内容为输入的命令）：

```
> test.py(5)test()
-> return int(name)
(Pdb) pp name
'cbekd'
(Pdb) n
ValueError: invalid literal for int() with base 10: 'cbekd'
> test.py(5)test()
-> return int(name)
(Pdb) 
```

