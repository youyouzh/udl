import os
import time

import requests
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data


# Return True if `x` (tensor or list) has 1 axis
def has_one_axis(x):
    return (hasattr(x, "ndim") and x.ndim == 1 or
            isinstance(x, list) and not hasattr(x[0], "__len__"))


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# 绘制多轴曲线图
def plot(x, y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    legend = [] if legend is None else legend

    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    if has_one_axis(x):
        x = [x]
    if y is None:
        x, y = [[]] * len(x), x
    elif has_one_axis(y):
        y = [y]
    if len(x) != len(y):
        x = x * len(y)
    axes.cla()
    for x, y, fmt in zip(x, y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    return plt


# ⽤于对多个变量进⾏累加
class Accumulator(object):
    def __init__(self, n):
        # n 表示累加的变量数量
        self.data = [0.0] * n

    def add(self, *args):
        # 分别对各变量进行累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator(object):
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        legend = [] if legend is None else legend
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()

    def show(self):
        plt.show()


# 时间计数器
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def sgd(params, learning_rate, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()


def torch_data_iter(features, labels, batch_size, is_train=True):
    """
    创建数据迭代器
    :param features: 特征
    :param labels: 标签
    :param batch_size: 批量大小
    :param is_train: 是否训练，true表示需要进行随机抽取
    :return: data_iter数据迭代器
    """
    dataset = (features, labels)
    dataset = data.TensorDataset(*dataset)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


load_array = torch_data_iter


def try_gpu(i=0):
    """
    如果存在，则返回gpu(i)，否则返回cpu()
    查看张量存放在什么设备： torch.rand(2, 2).device
    在GPU上创建张量： torch.ones(2, 3, device=torch.device('cuda')
    多个显卡上的张量不能直接运算，需要复制在同一个显卡上才能运算： Y.cuda(1)
    可以将Sequential转到GPU上，net.to(device=try_gpu())
    :param i 第几个显卡
    :return torch.device
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# 下载d2l数据集
def download_d2l_data(filename, cache_dir=None):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    cache_dir = cache_dir if cache_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filepath = os.path.join(cache_dir, filename)
    if os.path.isfile(filepath):
        print('The file is exist, do not need download: {}'.format(filepath))
        return filepath
    download_url = 'http://d2l-data.s3-accelerate.amazonaws.com/' + filename
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    print(f'正在从{download_url}下载{filename}...')
    response = requests.get(download_url, stream=True, verify=True)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    return filepath


def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    fig_size = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes
