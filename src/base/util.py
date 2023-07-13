from matplotlib import pyplot as plt


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
