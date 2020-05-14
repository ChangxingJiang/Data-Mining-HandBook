# Day-11 : Python的Excel操作：Openpyxl

> **作者**：长行
>
> **时间**：2020.05.13

Openpyxl是一个用于读写Excel2010文件的Python第三方包。相较于xlrd，虽然没有与pandas的有效交互，但是在样式设置上的功能更加强大。其pip安装命令如下：

```
pip install openpyxl
```

Openpyxl的整体逻辑是：首先通过打开或创建的方法，实例化Excel的工作簿文件（Workbook类）；接着再通过打开或创建的方法，实例化Excel的工作表（Worksheet类）；然后通过坐标或其他方式定位到一个或多个单元格执行操作；最后，将操作结果存入文件中。

## 工作簿层级操作

Openpyxl所有关于工作表的操作都需要在实例化的工作簿（openpyxl.workbook.workbook.Workbook）中进行，因此实例化

#### 实例化工作簿

工作簿的实例化有两种形式，分别为创建新的工作簿和打开已有的工作簿，就相当于新建Excel文件和打开已有的Excel文件。

创建新的工作簿：

```python
from openpyxl import Workbook
wb = Workbook()
```

打开已有工作簿：

```python
from openpyxl import load_workbook
wb = load_workbook('test.xlsx')
```

#### 存储工作簿

实例化在工作簿，无论是新建的还是打开的，在没有保存到文件之前，都只存在于内存中，一旦程序关闭就会消失，只有执行了save方法后内存中的工作簿才会被保存到本地文件。

```python
wb.save('test.xlsx')
```

## 工作表层级操作

Openpyxl所有关于行列的操作都需要在实例化的工作表（openpyxl.worksheet.worksheet.Worksheet）中进行。

工作表的实例化也有两种形式，分别为创建新的工作表和打开已有的工作簿，就相当于在Excel中创建新的工作表和选中已有的工作表。

#### 实例化工作表

##### 创建新的工作表

Openpyxl中使用create_sheet方法在工作簿中创建工作表，create_sheet方法有两个参数，第一个参数为创建工作表的名称，第二个为创建工作表的位置（若不填则默认创建为最后一个工作表）。

```python
ws1 = wb.create_sheet("Mysheet")  # 创建为最后一个工作表
ws2 = wb.create_sheet("Mysheet", 0)  # 创建为第一个工作表
ws3 = wb.create_sheet("Mysheet", -1)  # 创建为倒数第二个工作表
```

##### 打开已有工作表

在Openpyxl中，工作表的名字直接以工作簿的键的形式存在，可以直接通过键来获得实例化的工作表；另外，也可以通过工作簿的active属性，直接打开正在被使用的工作表，即打开Excel文件中首先打开的工作表。

```python
ws1 = wb["Sheet1"]  # 以键的形式打开工作表
ws2 = wb.active  # 打开正在被使用的工作表
```

#### 获取工作表名称列表

在Openpyxl中，可以使用```Workbook.sheetname```属性查看工作簿中所有工作表的名称。

```python
print(wb.sheetnames)  # 输出值: ['Mysheet1', 'Sheet1', 'Mysheet2', 'Mysheet']
```

#### 遍历工作表

在Openpyxl中，可以直接通过遍历工作簿的方式遍历工作簿中的所有工作表。

```python
for sheet in wb:
    print(sheet.title, end=" ")  # 输出值: Mysheet1 Sheet1 Mysheet2 Mysheet 
```

## 单元格层级操作

所有对单元格内容、格式的操作，都需要在实例化的单元格（<class 'openpyxl.cell.cell.Cell'>）中进行，实例化单元格的过程就相当于在Excel中选中单元格的过程。

在新建的工作表中，是不包含任意单元格的

#### 实例化单元格

Openpyxl中，单元格名称（例如A1）可以直接作为工作表的键获得实例化的单元格；另外，也可以通过```Worksheet.cell()```方法来获取实例化的单元格。

```python
cell1 = ws4["A2"]
cell2 = ws4.cell(row=1, column=1)
```

#### 实例化多个单元格

Openpyxl中，可以使用切片方式获取一定范围内的单元格实例列表，这个切片可以是从单元格到单元格，也可以是从行到行或从列到列；另外，也可以使用```Worksheet.rows```属性和```Worksheet.columns```属性来遍历文件。

下面我们以3×3的工作表为例，展示各种方法获取的单元格情况。

##### 切片

```python
for x in range(1, 4):
    for y in range(1, 4):
        ws4.cell(row=x, column=y)

cell_range_1 = ws4['A1':'C2']
cell_range_2 = ws4['A']
cell_range_3 = ws4['A:B']
cell_range_4 = ws4[2]
cell_range_5 = ws4[2:3]

print(cell_range_1)  # 输出元组包含：A1、B1、C1、A2、B2、C2
print(cell_range_2)  # 输出元组包含：A1、A2、A3
print(cell_range_3)  # 输出元组包含：A1、A2、A3、B1、B2、B3
print(cell_range_4)  # 输出元组包含：A2、B2、C2
print(cell_range_5)  # 输出元组包含：A2、B2、C2、A3、B3、C3
```

##### 遍历

```python
print(tuple(ws4.rows))  # 输出顺序：(A1,B1,C1),(A2,B2,C2),(A3,B3,C3)
print(tuple(ws4.columns))  # 输出顺序：(A1,A2,A3),(B1,B2,B3),(C1,C2,C3)
```

#### 读写单元格的值

在Openpyxl中，单元格对象的```value```属性就是单元格的值，通过读取和设置单元格的```value```属性可以实现单元格的读写。

```python
cell1.value = 35
print(ws4["A2"].value)  # 输出值:35
```

#### 读取单元格坐标

在Openpyxl中，单元格对象的```row```属性是单元格的行号，```column```属性是单元格的列号，```column_letter```是单元格的列的字母，```coordinate```是单元格的坐标名。

```python
# cell1为A2单元格
print(cell1.row)  # 输出值: 2
print(cell1.column)  # 输出值: 1
print(cell1.column_letter)  # 输出值: A
print(cell1.coordinate)  # 输出值
```

#### 设置单元格类型

在Openpyxl中，单元格对象的```data_type```属性即为单元格类型。其中数值、百分比、货币、分数、科学计数5种单元格格式均为n，日期和时间的格式均为d，字符格式为s。例如：

```python
ws4["B1"] = 35
ws4["B2"] = "我是字符串"
ws4["B3"] = datetime.datetime(2020, 5, 13)
print(ws4["B1"].data_type)  # 输出值: n
print(ws4["B2"].data_type)  # 输出值: s
print(ws4["B3"].data_type)  # 输出值: d
```

