# Windows的Path环境变量
> **作者**：长行\
> **时间**：2020.05.01

无论在搭建Python还是Java的环境时，都需要设置Windows的Path环境变量，那么Path环境变量究竟起到什么样的作用呢？
首先，当我们没有在Path环境变量中设置Python的路径时，执行python的任意命令都会得到如下结果。
```
C:\Users\Changxing>python --version
'python' 不是内部或外部命令，也不是可运行的程序或批处理文件。
```

然而当我们在Path环境变量中设置了Python的路径后，则会得到如下结果，而这就是Path环境变量的作用。
```
C:\Users\Changxing>python --version
Python 3.8.2
```

在“命令提示符”中执行的命令（例如上例中的“Python”），实际上都是执行的名为该命令的可执行文件（即Python路径下的python.exe文件）；而Path环境变量中所有的路径，则是系统检索是否存在名为该命令的可执行文件的范围。
因此，当我们在Path环境变量中添加了Python的路径后，系统就可以在Python的路径中检索到与python命令同名的可执行文件来运行了。

下面我们详细介绍一下Python环境。例如，我们将Python安装在了D盘的Python38_64文件夹下，此时我们设置的环境变量为：
```
D:\Python38_64\
D:\Python38_64\Scripts\
```

在“D:\Python38_64\”的路径中，我们可以找到“python.exe”；也就是说，我们在命令提示符中运行的python命令实际上是执行的这个可执行文件。
在“D:\Python38_64\Scripts\”的路径中，我们可以找到“pip.exe”、“jupyter-notebook.exe”（若安装了jupyter）等可执行文件；也就是说，我们在命令提示符中运行的pip、jupyter-notebook等命令实际上就是执行的这些可执行文件。
因此，当我们把Path环境变量中的Python路径修改为另一个版本的Python路径后，在命令提示符中运行的python、pip等一系列命令时所运行的可执行文件也将变为修改后版本的Python路径中的可执行文件。
