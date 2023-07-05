# Jupyter安装和使用

## Jupyter概述

### Jupyter简介

Jupyter Notebook是一个开源Web应用程序，允许您创建和共享包含实时代码，方程式，可视化效果和叙述文本的文档。用途包括：数据清理和转换，数值模拟，统计建模，数据可视化，机器学习等

Jupyter Notebook是一个交互式的笔记本，支持运行超过40种编程语言,Jupyter Notebook可以通过网页的形式打开，在网页页面中直接编写代码和运行代码，代码的运行结果也会直接在代码块下面进行显示。如果在编程过程中需要编写说明文档相关信息，可以使用Markdown直接进行编写，便于作及时的说明和解释。

### Jupyter组成部分

- 网页应用： 基于网页形式的、结合了编写说明文档、数学公式、交互计算和其他富媒体形式的工具，实现各种功能
- 文档： 后缀名为`.ipynb`的`JSON`格式文件，可以导出为：HTML、LaTeX、PDF等格式
- 主要特
  - 编程时具有语法高亮、缩进、tab补全的功能以及各种快捷键可供使用
  - 可直接通过浏览器运行代码，同时在代码块下方展示运行结果
  - 以富媒体格式展示计算结果。富媒体格式包括：HTML，LaTeX，PNG，SVG等
  - 对代码编写说明文档或语句时，支持Markdown语法
  - 支持使用LaTeX编写数学性说明

## Jupyter安装配置

可以通过conda和pip来进行安装。使用命令`pip install jupyter`。2023新版的conda中已经自带有`jupyter`，可以开始菜单搜索找到应用。

```shell
#安装ipykernel
conda install ipykernel
#写入环境
python -m ipykernel install  --name pytorch --display-name "Pytorch for Deeplearning"
```

找到应用后，可以直接启动`jupyter notebook`服务，默认会在<http://localhost:8888/tree>打开网站，显示Notebook的主页面。也可以在命令行启动：

- 自定义端口启动： `jupyter notebook --port <port_number>`
- 启动服务器但不打开浏览器: `jupyter notebook --no-browser`
- 生成配置文件： `jupyter notebook --generate-config`
- 
常规的情况下，Windows和Linux/macOS的配置文件所在路径和配置文件名如下所述：

- Windows系统的配置文件路径：`C:\Users\<user_name>\.jupyter\`
- Linux/macOS系统的配置文件路径：`/Users/<user_name>/.jupyter/` 或 `~/.jupyter/`
- 配置文件名：`jupyter_notebook_config.py`

配置文件内容：

```conf 
c.NotebookApp.ip = 'XX.XX.XX.XX'
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.password = u'sha1:XXXXXXXXXXXXXXXX'
c.ContentsManager.root_dir = 'D:\\work\\jupyter'
```

启动时会打开一个命令窗口，如果关掉这个窗口则会结束服务。

## PyCharm配置Jupyter

PyCharm-2019后的专业版都支持Jupyter，不过需要配置服务器，在`File -> Setting`中搜索`Jupyter`，找到`Jupyter Servers`，并配置服务器地址，比如：http://localhost:8888。

简单一些的话可以直接使用PyCharm的托管服务器，都不用启动Jupyter服务。

## VSCode配置Jupyter

VSCode也可以支持Jupyter，需要安装插件：`Python`，`Jupyter`，并配置相关的python执行路径以及Jupyter服务器。

- 进行python环境配置：File -> Preferences -> Settings，搜索"python.pythonPath"，将其指向正确的python安装位置
- 测试Python: 在VSCode编辑器中新开文件，输入 print("Hello World!")，按快捷键Ctrl+Shift+B，然后终端就会输出Hello World
- 在VSCode底部任务栏找到Jupyter Notebook，点击Launch，弹出Jupyter Notebook界面
- 在Jupyter Notebook界面中新建Python文件，在文件内输入print("Hello World")，然后点击Run，终端就会输出Hello World

## Jupyter使用。

Notebook file 是由一个个cell构成的，有三种类型 code、raw 与markdown cell。(cell: 一对In Out会话被视作一个代码单元, 每个代码单元可以单独执行)。

在编辑Python Code Cell 时，相应的代码补全，语法高亮显示，错误信息提示及快速修复等等功能与编辑标准的Python文件一样都是支持的。

文件创建完成以后，在 PyCharm 里可以直接运行了。PyCharm 提供了多种运行方式，既可以单独运行一个 Cell, 也可以一次全部运行。不仅可以在同一页面查看运行结果，还可以查看变量详情。
