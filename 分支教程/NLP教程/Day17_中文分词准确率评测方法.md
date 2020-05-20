# Day17 : 隐马尔可夫模型

> **作者**：长行
>
> **时间**：2020.05.10

准确率可以用来衡量中文分词的准确程度，包含一系列的评测指标。

下面我们用一个例子来解释这些评测指标：

已知某天某小学共100名小学生中有99名小学生都佩戴了红领巾，只有1名小学生忘记了。学校为检查小学生是否佩戴红领巾，安装了自动拍照检查报警仪。

## 1

严格来说，当预测数量与答案数量相等时，准确率的计算公式为：

准确率 = 判断正确的次数 / 总测试次数

使用刚才的例子，若自动检查报警仪是个假冒伪劣产品，无论是否佩戴红领巾，都不会报警。此时仪器的准确率为99/100=99%，这样的结果我们显然是不能接受的。

## 2

由此我们引入精确率和召回率的概念。

首先，我们将没有佩戴红领巾的情况定义为正类(P)，相应的，佩戴红领巾的情况即为负类(N)。

精确率是指所有准确预测为正类的结果占所有预测为正类的结果的比例，也就是所有“机器认定的没有佩戴且实际没有佩戴红领巾的小学生”占“机器认定的没有佩戴红领巾的小学生”的比例。

召回率是指所有准确预测为正类的结果占所有实际为正类的结果的比例，也就是所有“机器认定的没有佩戴且实际没有佩戴红领巾的小学生”占“实际没有佩戴红领巾的小学生”的比例。

下面我们引入混淆矩阵来更好的解释；混淆矩阵是一种用来衡量分类的混淆程度的图表。

![img](E:\【有道云笔记】\weixinobU7VjnGIqORtayCj-b7o8rEdBNc\8ec105ae4bdc4c24b56523aee105fb66\1583891885(1).png)

> TP：预测结果为P，正确答案也为P
>
> FP：预测结果是P，正确答案为N
>
> TN：预测结果为N，正确答案为P
>
> FN：预测结果为N，正确答案也为N

此时：

精确率的计算公式为：P = TP / (TP+DF)；这台假冒伪劣产品的精确率为0。

召回率的计算公式为：R = TP / (TP+FN)；这台假冒伪劣产品的召回率也为0。

## 3

因为在系统排名时，人们习惯只使用一个指标。因此我们引入了一个精确率和召回率调的调和平均F1值，作为一个综合性的指标。F1值的计算公式为：

F1 = 2×P×R / (P+R)

此时必须精确率和召回率同时较高，才能得到较高的F1值。

## 4

下面我们将这种评测方法引入到中文分词领域。

我们将标准答案中每个单词在文本中的起止位置区间坐标构造为集合A，分词结果中的坐标构造为集合B。此时：

```
A = TP∪FN
B = TP∪FP
TP = A∩B
P = |A∩B| / |B|
R = |A∩B| / |A|
```

我们使用pyhanlp中的函数实现：

```python
import re
from pyhanlp import *

def to_region(segmentation: str) -> list:
    """ 将分词结果转换为区间 """
    region = []  # 结果区间
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region

def prf(gold: str, pred: str) -> tuple:
    """
    计算P、R、F1
    :param gold: 标准答案文件
    :param pred: 分词结果文件
    """
    A_size, B_size, A_cap_B_size = 0, 0, 0
    with open(gold, encoding='utf-8') as gd, open(pred, encoding='utf-8') as pd:
        for g, p in zip(gd, pd):
            A, B = set(to_region(g)), set(to_region(p))
            A_size += len(A)
            B_size += len(B)
            A_cap_B_size += len(A & B)
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r)
```

> 学习参考文献：《自然语言处理入门》(何晗)：2.9.1-2.9.7