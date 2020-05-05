# 第7天 : Python常用内置函数和模块
> **作者**：长行\
> **时间**：2020.05.05

## 常用内置函数
Python到目前为止（3.8.2）共计提供了68个内置函数，这些函数可以在任何条件下被直接调用。
> 官方文档地址：https://docs.python.org/zh-cn/3.8/library/functions.html

在官方文档中，各个内置函数依据字母顺序进行排列，学习查询相对不便。下面，我们依据内置函数的类型，对常用的内置函数进行整理学习。\
在学习的过程中，我们不需要记住每一个内置函数的名字及其对应功能，但是尽可能在脑海中形成一个什么样的功能可以直接通过内置函数实现的印象。在实际操作时，可以再回过头来查询内置函数的名称及其具体用法，常用的内置函数多用几次也就逐渐熟悉了。\
因为我们当前还没有学习类与对象的相关知识，因此“对象反射相关”部分暂为选学。

### 数据类型相关
数据类型相关的内置函数包含与数值变量相关、与序列变量相关和与容器类变量相关三种。
#### 数值变量相关
| 函数 | 功能 |
| ------ | ------ |
| bool([x]) | 使用真值测试过程来返回一个布尔值，即True或False |
| complex([real[, imag]]) | 返回一个新的complex对象，即复数对象 |
| float([x]) | 返回从数字或字符串生成的浮点数 |
| int([x]) | 返回一个基于数字或字符串x构造的整数对象；若未给出参数则返回0 |
| int(x,base=10) | 返回一个基于字符串x构造的以base为进制的整数对象 |
#### 序列变量相关(列表/元组)
| 函数 | 功能 |
| ------ | ------ |
| enumerate(iterable, start=0) | 返回一个枚举对象，其中包括计数值和通过迭代iterable获得值 |
| filter(function, iterable) | 用iterable中，带入函数function返回为True的元素构成一个新的迭代器 |
| format(value[,format_spec]) | 将value转换为format_spec控制的“格式化”表示 |
| len(s) | 返回对象的长度（元素个数） |
| list([iterable]) | 返回一个list对象，即可变序列类型 |
| map(function, iterable, ...) | 返回一个将function应用于iterable中每一项并输出器结果的迭代器 |
| reversed(seq) | 返回一个反向的iterator |
| slice(stop) | 返回一个由`range(stop)`所指定索引集的slice对象 |
| slice(start,stop[,step]) | 返回一个由`range(start,stop,step)`所指定索引集的slice对象 |
| sorted(iterable, \*, key=None, reverse=False) | 将iterable中的项目重新排序为新的列表 |
| tuple([iterable]) | 返回一个新的tuple对象 |
| zip(\*iterables) | 创建一个聚合了来自每个可迭代对象中的元素的迭代器 |
#### 序列变量相关(字符串/字节序列)
| 函数 | 功能 |
| ------ | ------ |
| ascii(object) | 返回一个可打印的字符串 |
| bytearray([source[,encoding[,errors]]]) | 返回一个新的bytes数组，即可变字节序列 |
| bytes([source[,encoding[,errors]]]) | 返回一个新的bytes对象，即不可变字节序列 |
| chr(i) | 返回Unicode编码为整数i的字符的字符串格式，例如`chr(97)`返回字符串 `'a'`，是ord()的逆函数 |
| memoryview(obj) | 返回一个依据参数创建的“内存视图”对象 |
| ord(c) | 对于表示单个Unicode字符的字符串，返回表它的Unicode编码，例如`ord('a')`返回整数`97`，是chr()的逆函数 |
| repr(object) | 返回包含一个对象的可打印的字符串 |
| str(object='') | 返回一个新的str对象 |
| str(object=b'',encoding='utf-8',errors='strict') | 返回一个新的str对象 |
#### 容器类变量相关(集合/字典)
| 函数 | 功能 |
| ------ | ------ |
| dict(\*\*kwarg) | 创建一个新的字典 |
| dict(mapping,\*\*kwarg) | 创建一个新的字典 |
| dict(iterable,\*\*kwarg) | 创建一个新的字典 |
| frozenset([iterable]) | 返回一个包含可选参数iterable中所有元素的新的frozenset对象 |
| set([iterable]) | 返回一个新的set对象，包含可选参与iterable中的所有元素 |

### 数学运算相关
数学运算相关的内置函数主要包括数学运算、逻辑运算和进制转换。
#### 数学运算
| 函数 | 功能 |
| ------ | ------ |
| abs(x) | 返回参数x的绝对值 |
| divmod(a,b) | 执行两个参数的除法并返回商和余数，类似于(a // b, a % b) |
| max(iterable,\*[,key,default]) | 返回可迭代对象中最大的元素 |
| max(arg1,arg2,&args[,key] | 返回参数中最大的元素 |
| min(iterable,\*[,key,default]) | 返回可迭代对象中最小的元素 |
| min(arg1,arg2,&args[,key] | 返回参数中最小的元素 |
| pow(base,exp[,mod]) | 返回base的exp次幂 |
| round(number[,ndigits]) | 返回nunmer四舍五入到小数点后ndigits位精度的值 |
| sum(iterable,/,start=0) | 从start开始自左向右对iterable的项求和并返回总值 |
#### 逻辑运算
| 函数 | 功能 |
| ------ | ------ |
| all(iterable) | 若iterable中所有元素为真，则返回True |
| any(iterable) | 若iterable中任意元素为真，则返回True |
#### 进制转换
| 函数 | 功能 |
| ------ | ------ |
| bin(x) | 将一个整数转变为一个前缀为'0b'的二进制字符串 |
| hex(x) | 将一个整数转变为一个前缀为'0x'的小写十六进制字符串 |
| oct(x) | 将一个整数转变为一个前缀为'0o'的八进制字符串 |
### 作用域相关
作用域相关的内置函数包括获以字典的形式获取局部变量和全局变量。

| 内置函数 | 功能 |
| ------ | ------ |
| globals() | 返回当前全局变量的字典 |
| locals() | 返回当前命名空间内局部变量的字典 |
### 迭代器相关
| 内置函数 | 功能 |
| ------ | ------ |
| iter(object[, sentinel]) | 返回一个iterator对象 |
| next(iterator[, default]) | 通过调用iterator的\_\_next()\_\_方法获取下一个元素（如果迭代器耗尽，则返回给定的default） |
| range(stop) | 返回一个从0开始，到stop结束的不可变的序列类型，一般用作循环 |
| range(start, stop[, step]) | 返回一个从start开始，到stop结束的，以step为步长的不可变的序列类型，一般用作循环 |
### 类与对象相关
对象、反射相关的内置函数主要包括：
#### 对象基本内容
| 函数 | 功能 |
| ------ | ------ |
| @classmethod | 把一个方法封装为类方法 |
| object | 返回一个没有特征的新对象 |
| property(fget=None, fset=None, fdel=None, doc=None) | 返回property属性 |
| @staticmethod | 把一个方法转换为静态方法 |
| super([type[,object-or-type]]) | 返回一个代理对象，它会将方法调用委托给type的父类或兄弟类 |
| type(object) | 返回对象的类型 |
| type(name,bases,dict) | 返回一个新的type对象 |
| vars([object]) | 返回模块、类、实例或任何其他具有\_\_dict\_\_属性的对象的\_\_dict\_\_属性 |
#### 对象关系
| 函数 | 功能 |
| ------ | ------ |
| isinstance(object,classinfo) | 若参数object是参数classinfo的实例或其子类则返回True |
| issubclass(class,classinfo) | 若class是classinfo的子类则返回True（类会被视作其自身的子类） |
#### 对象属性
| 函数 | 功能 |
| ------ | ------ |
| delattr(object,name) | 如果对象允许，则删除函数中指定的属性 |
| dir([object]) | 如果没有实参，则返回当前本地作用域中的名称列表；如果有实参，则返回该对象的有效属性列表 |
| getattr(object,name[,default]) | 返回对象指定属性的值 |
| hasattr(object,name) | 如果name是对象object的属性之一的名称，则返回True |
| setattr(object,name,value) | 将值value赋予对象object中的名为name的属性 |
#### 对象调用
| 函数 | 功能 |
| ------ | ------ |
| callable(object) | 如果参数object是可调用的就返回True |
### 其他
#### 编译执行
| 函数 | 功能 |
| ------ | ------ |
| compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1) | 将source编译成代码或AST对象；代码对象可以被exec()或eval()执行 |
| eval(expression[, globals[, locals]]) | 将字符串类型的代码(expression)执行并求值 |
| exec(expression[, globals[, locals]]) | 将字符串类型的代码(expression)执行并返回None |
#### 控制台输入/输出
| 函数 | 功能 |
| ------ | ------ |
| input([prompt]) | 从控制台输入中读取一行，将其转换为字符串并返回 |
| print(\*objects, sep=' ', end='\n', file=sys.stdout, flush=False) | 将objects打印到控制台中，以sep分隔并在末尾加上end |
#### 内存相关
| 函数 | 功能 |
| ------ | ------ |
| hash(object) | 返回该对象的哈希值 |
| id(object) | 返回对象的“标识值”，在对象的生命周期中是唯一且恒定的 |
#### 文件相关
| 函数 | 功能 |
| ------ | ------ |
| open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None) | 打开文件并返回对应的文件对象 |
#### 模块导入
| 函数 | 功能 |
| ------ | ------ |
| \_\_import\_\_(name, globals=None, locals=None, fromlist=(), level=0) | 导入模块包 |
#### 帮助系统
| 函数 | 功能 |
| ------ | ------ |
| helo([object]) | 启动内置的帮助系统（主要用于IDLE） |
#### 代码调试
| 函数 | 功能 |
| ------ | ------ |
| breakpoint(\*args, \*\*kws) | 函数会在调用时进入调试器中 |












