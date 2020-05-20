# Day18 : 基于HanLP的隐马尔可夫模型实现

> **作者**：长行
>
> **时间**：2020.05.10

隐马尔科夫模型的介绍已在第16天的学习过程中记录。

HanLP实现了基于隐马尔可夫模型的中文分词器HMMSegmenter，其中包含训练接口和分词接口。

其实现代码如下：

```python
from pyhanlp import *
from HanLP_Book.tests.test_utility import ensure_data

FirstOrderHiddenMarkovModel = JClass('com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel')  # HanLP中隐马尔可夫模型分词Java模块
HMMSegmenter = JClass('com.hankcs.hanlp.model.hmm.HMMSegmenter')  # HanLP中隐马尔可夫模型训练Java模块

sighan05 = ensure_data('icwb2-data', 'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip')  # HanLP中训练语料压缩包
msr_train = os.path.join(sighan05, 'training', 'msr_training.utf8')  # HanLP中MSR训练语料路径

def train(corpus, model):
    segmenter = HMMSegmenter(model)
    segmenter.train(corpus)
    return segmenter

if __name__ == "__main__":
    segmenter = train(msr_train, FirstOrderHiddenMarkovModel())  # 训练隐马尔可夫模型
    print(segmenter.segment("商品和服务"))  # 使用隐马尔可夫模型分词
```

运行结果

```
[商品, 和, 服务]
```

下面我们来稍微探索一下HanLP的实现原理。大体上包括如下流程：

1. 将标注集{B,M,E,S}映射为连续的整形id。
2. 将字符串形式的字符映射为整型形式的字符。
3. 将语料库的格式，转换为(x,y)二元组。
4. 依据整型形式的标注集和字符，使用二元组格式的语料库进行训练，

> 学习参考文献：《自然语言处理入门》(何晗)：4.6