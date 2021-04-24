# Python文本整理案例：直播弹幕数据清洗

> **作者**：长行
>
> **时间**：2020.05.18

## 任务需求

清洗json格式的弹幕数据（以“聆听丶芒果鱼直播间时间切片弹幕.json”为例），具体要求如下：

 * 提取出所有弹幕类型（列表中的第1个元素）为”NM“的弹幕的内容（列表中的第4个元素）取出，存为一条弹幕为一行的txt格式文件。
 * 剔除弹幕中的不可见字符（包括制表符、换行符等）
 * 将弹幕中的所有英文字符转换为小写
 * 将弹幕中的所有全角字符转换为半角字符
 * 将弹幕中的所有繁体字转换为简体字
 * 将弹幕中大于等于4个的、连续的、相同的中文汉字合并为3个（例如将”啊啊啊啊啊啊“替换为”啊啊啊“）
 * 将弹幕中大于等于4个的、连续的、相同的英文字母或数字合并为3个（例如将”666666“替换为”666“）
 * 将弹幕中大于等于3个的、连续的、相同的标点符号合并为2个（例如将“???”替换为”??“）

> 弹幕的Json文件位于当前目录下

## 实现方法

基于以上需求，我们逐个确定实现方法。

#### 剔除不可见字符

根据对弹幕内容中包含的不可见字符的分析，我们主要需要剔除制表符(\t)、回车符(\r)和换行符(\n)。

```python
source = source.replace("\t", "").replace("\r", "").replace("\n", "")
```

#### 替换英文大小写

将所有英文字符转换为小写只需要使用字符串自带的方法即可。

```python
source = source.upper()  # 将字符串中的英文全部转换为大写
source = source.lower()  # 将字符串中的英文全部转换为小写
```

#### 将全角字符转换为半角字符

我们遍历弹幕字符串中的每个字符，依据Unicode编码将全角字符转换为半角字符。

```python
def full_width_to_half_width(string):
    """ 将全角字符转化为半角字符
    
    :param string: <str> 需要转化为半角的字符串
    :return: <str> 转化完成的字符串
    """
    result = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        result += chr(inside_code)
    return result
```

#### 将繁体字转换为简体字

因为繁简转换涉及到繁简分歧词等多种情况，所以我们不宜直接按字转换，而更应该按词转换。因此我们使用其他人造的轮子，通过HanLP的convertToSimplifiedChinese函数将弹幕字符串中的繁体字转换为简体字。

```python
from pyhanlp import HanLP
source = HanLP.convertToSimplifiedChinese(source)
```

#### 合并连续相同的字符

我们使用包含反向引用的正则表达式来匹配连续且相同的字符，又考虑到每个句子当前可能包含多个不同的连续且相同的字符，因此使用findall方法来进行匹配。

> 反向匹配是指正则表达式中的"\1"，“\1”可以用来指代第一个被“()”定义的子表达式。

```python
for chinese_character in re.findall(r"([\u4e00-\u9fa5])\1{3,}", source):
    source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
for chinese_character in re.findall(r"([A-Za-z])\1{3,}", source):
    source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
for chinese_character in re.findall(r"([0-9])\1{3,}", source):
    source = re.sub("[" + chinese_character[0] + "]{3,}", chinese_character * 3, source)
```

（以上三个循环的正则表达式可以合并并简化为一个循环）

类似的，连续且相同的标点符号的处理方法如下。

```python
PUNCTUATION_LIST = [" ", "　", ",", "，", ".", "。", "!", "?"]  # 样例标点符号列表
punctuation_list = "".join(PUNCTUATION_LIST)
for match_punctuation in re.findall("([" + punctuation_list + "])\\1{2,}", source):
    source = re.sub("[" + match_punctuation[0] + "]{2,}", match_punctuation * 3, source)
source = re.sub("-{2,}", "---", source)  # 处理特殊的短横杠
```

#### 结果输出

在完成弹幕内容的清洗后，我们可以将临时变量中的数据存储到文件中。

```python
with open("时间切片弹幕(清洗后).txt", "w+", encoding="UTF-8") as file:
    file.write("\n".join(barrage_list))
```

> 完整源代码位于当前目录下



