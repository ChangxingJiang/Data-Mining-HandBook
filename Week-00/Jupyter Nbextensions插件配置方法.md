# Jupyter Nbextensions插件配置方法
> **作者**：长行\
> **时间**：2020.05.04

## Nbextensions简介
Nbextensions软件包是Jupyter非官方扩展包的集合，可以为Jupyter提供很多实用的功能。
这些扩展包大部分为Javascript编写，在运行Jupyter浏览器时被本地加载。
> 文档地址：https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html

## Nbextensions扩展包插件的安装方法
关闭Jupyter，在命令提示符(cmd)中执行如下命令：
```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
```
重新打开Jupyter即可看到已经安装成功的Nbextensions扩展包插件。

## nbextensions插件说明
* (some) LaTeX environments for Jupyter : 支持更多的Latex元素
* 2to3 Converter : 将Python2代码转换为Python3代码的工具
* Addbefore : 在当前单元格前加入空单元格的功能
* Autopep8 : 代码格式化的工具(需要安装autopep8包)
* Autosavetime : 可设置自动保存的时间间隔
* Autoscroll : 设置屏幕自动滚动阈值的工具
* Cell Filter : 依据标签过滤单元格的工具
* Code Font Size : 设置代码字体大小的工具
* Code preffify : 代码美化的工具(需要安装yapf包)
* Codefolding : 增加折叠代码的功能(包括Python依据缩进折叠、其他代码依据方括号折叠、依据注释折叠)
* Codefolding in Editor : 
* CodeMirror mode extensions : 
* Collapsible Headings : 存储markdown标题的折叠情况(在下一次打开的时候重新加载这点状态)
* Comment/Uncomment Hotkey : # 增加注释/取消注释的热键
* contrib_nbextensions_help_item : 添加帮助菜单
* datestamper : 将当前日期和时间粘贴到单元格中的工具
* Equation Auto Numbering : 公式自动编号工具
* ExecuteTime : 记录上一次运行代码单元的时间以及运行花费的时间
* Execution Dependencies : 记录单元格之间依赖关系的工具
* Exercise : 隐藏/显示单元格的功能(可以与Exercise2同时使用)
* Exercise2 : 隐藏/显示单元格的功能(可以与Exercise同时使用)
* Export Embedded HTML : 将notebook导出为HTML文件的工具
* Freeze : 提供单元格锁定、只读相关功能
* Gist-it : 一键发布Github的gist功能
* Help panel : 在工具栏中增加显示帮助说明的按钮
* Hide Header : 增加隐藏标题栏、菜单栏和工具栏的功能
* Hide input : 在工具栏中增加隐藏指定代码单元的功能
* Hide input all : 在工具栏中增加隐藏所有代码单元的功能
* Highlight selected word : 高亮显示被选中的对象以及所有代码单元中该对象的所有实例
* hightlighter : 在工具栏中增加荧光笔功能，可以高亮显示框选的内容
* Hinterland : 自动补全功能
* Initialization cells : 在工具栏中增加运行所有被框选为“初始化代码单元”的代码单元
* insort formatter : 排序代码单元中导入的包(需要安装isort包)
* jupyter-js-widgets/extension : 
* Keyboard shortcut editor : 创建/修改/删除Jupyter的快捷键
* Launch QTConsole : 使用QTConsole内核
* Limit Output : 显示代码单元输出的文本或HTML的字符数
* Live Markdown Preview :
* Load Tex macros : 
* Move selected cells: 在工具栏中增加上下移动单元格的功能
* Navigation-Hotkeys : 增加用于导航的快捷键
* Nbextensions dashboard tab :
* Nbextensions edit menu item : 
* nb Translate : 在工具栏中增加语言翻译工具 
* Notify : 增加在Jupyter内核空闲时的通知提示功能(适合运行很耗时的情况)
* Printview : 在工具栏中增加将当前notebook输出为html或pdf的功能
* Python Markdown : 允许使用{{x}}的方法来直接输出结果
* Rubberband : 允许进行多个单元的选择
* Ruler : 增加标尺功能
* Ruler in Editor : 
* Runtools : 在工具栏中增加了更多运行各个单元的方法
* Scratchpad : 增加便笺式单元，可以针对当前内核执行代码，而无需修改notebook中的内容
* ScrollDown : 自动在选中单元格时向下滚动
* Select CodeMirror Keymap : 使用CodeMirror获取键盘映射，可以禁用非Jupyter的其他浏览器快捷键
* SKILL Syntax : CodeMirror的Skill模块
* Skip-Traceback : 跳过报错的路径追踪，只显示错误名称和类型的摘要
* Snippets : 在工具栏中增加添加指定代码单元的功能
* Snippets Menu : 在菜单栏中增加可自定义的菜单栏，用以插入代码片段
* spellchecker : 拼写检查，高亮显示拼写错误的单词
* Split Cells Notebook : 增加拆分单元格的命令
* Table of Contents(2) : 增加浮动目录功能
* table_beautifier : 美化输出的单元格
* Toggle all line numbers : 在工具栏中增加一个控制所有单元格行号是否显示的工具
* Tree Filter : 在Jupyter笔记本文件树页面中增加按文件名过滤的功能
* Variable Inspector : 在工具栏中增加变量检查的功能
* zenmode : 增加Zenmode功能扩展
