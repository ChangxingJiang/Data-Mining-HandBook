# Day03 : 语言元素
> **作者**：长行\
> **时间**：2020.05.04

## 变量
### 使用变量
此前，我们已经了解了控制台输入和输出的方法，通过以下代码，我们可以实现了读取用户想到的数字并输出。

```python
number = input("你想到了什么数字? ")
print("你想到的数字是:", number)
```
在这段代码中：第一行我们使用input()函数从控制台读取了用户输入的内容，并将其存入了名为number的变量中，而在第二行中，我们使用print()函数将number变量的值输出到了控制台中。\
我们通常将第一行对number变量的操作称为“变量赋值”，将第二行对number变量的操作称为“引用变量”。
### 变量类型
接着，我们想要返回用户想到数字的平方。

```python
number = input("你想到了什么数字? ")
print("你想到数字的平方是:", number * number) # 这一行会报错
```
在这段代码中，我们使用number*number来计算number变量的平方。但是不但不能得到我们想要的结果，返回还会令程序报错。\
这是因为变量是有类型的区别的。我们从控制台读取的用户想要的数字是一个字符串，而对于一个字符串来说，它自己乘自己是无法被计算的。因此，我们需要在乘方之前，先将这个字符串转化为数字，这样就可以计算了。

```python
number = input("你想到了什么数字? ")
print("你想到数字的平方是:", int(number) * int(number))
```
在这段代码中：我们使用int()函数，先将字符串类型的number变量转换为数字类型的变量，而后再计算平方，就不再会报错，可以得到我们想要的结果了。\
由此可知，不同类型的Python变量能够进行的运算是不一样的，它们之间也是可以进行转换的。
在Python中，我们常见的变量类型包括：
* 整型(int)：任意大小的整数
* 浮点型(float)：任意长度的小数
* 字符串型(str)：任意长度的文本
* 布尔型(bool)：布尔型变量只有True和False两种值
* 复数(complex)：任意长度的复数(实部+虚部)

#### 检查变量的类型
具体的，我们可以使用type()来检查变量的类型。
```python
var_1 = 3  # 整型(int)
var_2 = 3.1415926  # 浮点型(float)
var_3 = 2 + 1j  # 虚数(complex)
var_4 = "Hello World"  # 字符串(str)
var_5 = True  # 布尔型(bool)
print(type(var_1))  # 返回值: <class 'int'>
print(type(var_2))  # 返回值: <class 'float'>
print(type(var_3))  # 返回值: <class 'complex'>
print(type(var_4))  # 返回值: <class 'str'>
print(type(var_5))  # 返回值: <class 'bool'>
```

#### 变量类型的转换
类似的，我们也可以使用int()、str()等方法将转换变量的类型。
```python
var = 3.1415926
print(type(int(var)),int(var))  # 返回值: <class 'int'> 3
print(type(str(var)),str(var))  # 返回值: <class 'str'> 3.1415926
```

变量赋值方法与基础运算详见：[常用变量类型与基础运算](https://github.com/Changxing97/Python-Data-mining-Tutorial/blob/master/Week-01/Day03/%E5%B8%B8%E7%94%A8%E5%8F%98%E9%87%8F%E7%B1%BB%E5%9E%8B%E4%B8%8E%E5%9F%BA%E7%A1%80%E8%BF%90%E7%AE%97.ipynb) 

### 变量命名
在给第一次赋值变量时，就命名了变量，在Python中，命名变量有如下规则：
* 变量名应由字母或下划线开头，并由字母、数字、下划线(_)组成
* 变量名不能与Python关键字同名，但可以包含关键字

通常来说，我们在命名变量名时，首字母不大写，不使用驼峰式，即thisIsVariable；而是使用下划线，即this_is_variable。（全局变量全部大写）

## 变量基本操作
下面，我们了解一下数值型变量和字符串变量的基本操作。
### 数值变量的基本操作
对于数值变量。我们首先，数值变量可以直接进行加减乘除的运算，也可以通过括号来调整计算顺序。
```python
a = 3
b = 5
print(a + b)  # 返回值: 8
print(b / a)  # 返回值: 1.6666666666666667
print((a + b) * b)  # 返回值: 40
```
有的时候，我们不需要得到小数形式的商，而是需要得到商的整数，也就是“取模”，可以使用“//”运算符；又或者只需要得到余数，也就是“取余”，可以使用“%”运算符。
```python
a = 3
b = 5
print(b // a)  # 返回值: 1
print(b % a)  # 返回值: 2
```
还有的时候，我们需要进行幂运算，可以使用“**”运算符。
```python
print(3 ** 5)  # 返回值: 243
```

### 字符串的基本操作
字符串的定义方法很简单，使用`"`和`'`都可以。
```python
var_1 = "Hello World"
var_2 = 'Hello World'
```
#### 字符串的四则运算
字符串也可以进行加法和乘法的计算，加法必须是字符串加字符串，乘法必须是字符串乘整数。
```python
var_1 = "Hello World"
var_2 = 'Hello'
print(var_1 + var_2)  # 返回值: Hello WorldHello
print(var_2 * 2)  # 返回值: HelloHello
```
#### 字符串长度
我们还可以通过len()函数来获取字符串的长度。
```python
var_1 = "Hello World"
var_2 = 'Hello'
print(len(var_1))  # 返回值: 11
print(len(var_2))  # 返回值: 5
```
#### 截取字符串
此外，有的时候我们需要获取字符串的某个部分，可以使用`[:]`来提取。
```python
var_1 = "Hello World"
print(var_1[3])
print(var_1[3:5])
```
（应注意序号是从0开始计算的，第n个字符的序号为n-1；序号范围是从左闭右开区间，第n个字符到第m个字符的序号为n-1到m）
