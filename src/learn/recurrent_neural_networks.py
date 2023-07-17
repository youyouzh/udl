"""
循环神经网络
"""
import torch

from base.util import plot


# 生成sin序列数据
def generate_sin_data(T=1000):
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3)).show()


if __name__ == '__main__':
    generate_sin_data()
