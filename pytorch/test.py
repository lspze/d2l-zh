import numpy as np
from matplotlib_inline import backend_inline
import matplotlib
from matplotlib import pyplot as plt

def numerical_lim(x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return 3 * x ** 2 - 4 * x

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(1, h):.5f}')
    h *= 0.1

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    # use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)  # x轴名称
    axes.set_ylabel(ylabel)  # y轴名称
    axes.set_xscale(xscale)  # x轴比例
    axes.set_yscale(yscale)  # y轴比例，如线性、对数等等
    axes.set_xlim(xlim)      # x轴上下限
    axes.set_ylim(ylim)      # y轴上下限
    if legend:
        axes.legend(legend)  # 设置图例
    axes.grid()
    
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
x = np.arange(0, 3, 0.1)
plt.subplot(2,1,1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.subplot(2,1,2)
m, n = [1,2,3], [3,4,5]
plt.plot(m,n)
plt.savefig('temp.png')