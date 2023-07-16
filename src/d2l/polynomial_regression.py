# 多项式回归
import math

import numpy as np
import torch
from torch import nn

from base.util import plt, Accumulator, Animator, torch_data_iter
from d2l.multilayer_perceptron import MultilayerPerceptron


def generate_polynomial_data(max_degree=20, n_train=100, n_test=100):
    """
    生成多项式数据集
    :param max_degree: 多项式的最大阶数
    :param n_train: 训练数据集大小
    :param n_test: 测试数据集大小
    :return:
    """
    true_w = np.zeros(max_degree)  # 分配大量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])   # 多项式系数

    features = np.random.normal(size=(n_train + n_test, 1))  # 生成特征集
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        # 使用阶乘函数进行系数缩放，避免⾮常⼤的梯度值或损失值
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    # labels的维度:(n_train+n_test,)
    labels = np.dot(poly_features, true_w)
    # 添加噪声
    labels += np.random.normal(scale=0.1, size=labels.shape)
    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype= torch.float32)
                                               for x in [true_w, features, poly_features, labels]]
    return true_w, features, poly_features, labels


def test_generate_polynomial_data():
    true_w, features, poly_features, labels = generate_polynomial_data()
    # 显示图像
    plt.scatter(features, labels, 1)
    plt.show()


class PolynomialRegression(MultilayerPerceptron):

    def __init__(self, w_shape):
        # 可以设置权重参数，也就是多项式的权重数，即多项式最高次幂
        self.w_shape = w_shape
        self.animator = None
        super().__init__()

    def define_model(self):
        self.net = nn.Sequential(nn.Linear(self.w_shape, 1, bias=False))
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)

    def evaluate_loss(self, data_iter):
        metric = Accumulator(2)
        for X, y in data_iter:
            out = self.net(X)
            y = y.reshape(out.shape)
            loss = self.loss_func(out, y)
            metric.add(loss.sum(), loss.numel())
        return metric[0] / metric[1]

    def train(self):
        train_data_iter = self.get_train_data_iter()
        test_data_iter = self.get_test_data_iter()
        self.animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                 xlim=[1, self.num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
        for epoch in range(self.num_epochs):
            self.train_epoch(train_data_iter)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                train_loss = self.evaluate_loss(train_data_iter)
                test_loss = self.evaluate_loss(test_data_iter)
                self.animator.add(epoch + 1, (train_loss, test_loss))
                print(f'epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}')


# 带权重衰减L2范数的模型
class WeightDecayPolynomialRegression(PolynomialRegression):

    def __init__(self, w_shape, l2_lambda):
        self.l2_lambda = l2_lambda   # L2范数系数
        super().__init__(w_shape)

    def calc_loss(self, y_hat, y):
        # 损失加上L2范数
        return self.loss_func(y_hat, y) + self.l2_lambda * torch.sum(self.net[0].weight.data.pow(2)) / 2


# 简化的权重衰减
class ConciseWeightDecayPolynomialRegression(WeightDecayPolynomialRegression):

    def __init__(self, w_shape, l2_lambda):
        super().__init__(w_shape, l2_lambda)
        # 设置SGD相关参数加入权重衰减
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)


def test_polynomial_regression(train_data_size, test_data_size, wight_size, num_epochs=500, l2_lambda=0.0):
    """
    :param train_data_size: 训练数据量
    :param test_data_size: 测试数据量
    :param wight_size: 训练权重数量（多项式阶数）
    :param num_epochs: 训练步数
    :param l2_lambda: l2范数系数
    """
    # 加载数据集
    true_w, features, poly_features, labels = generate_polynomial_data()
    polynomial_regression = WeightDecayPolynomialRegression(wight_size, l2_lambda)
    # polynomial_regression = ConciseWeightDecayPolynomialRegression(wight_size, l2_lambda)
    # 设置超参数
    polynomial_regression.preset_hyper_param(num_epochs=num_epochs, learning_rate=0.1, batch_size=min(10, labels.shape[0]))

    # 三阶多项式函数拟合(正常)
    train_data_iter = torch_data_iter(poly_features[:train_data_size, :wight_size],
                                      labels[:train_data_size].reshape(-1, 1),
                                      polynomial_regression.batch_size)
    test_data_iter = torch_data_iter(poly_features[:test_data_size, :wight_size],
                                     labels[:test_data_size].reshape(-1, 1),
                                     polynomial_regression.batch_size, is_train=False)
    polynomial_regression.set_data_iter(train_data_iter, test_data_iter)
    polynomial_regression.train()
    polynomial_regression.animator.show()
    print('true w: {}, train w: {}'.format(true_w, polynomial_regression.net[0].weight.data.numpy()))


if __name__ == '__main__':
    # 三阶多项式函数拟合(正常)
    # test_polynomial_regression(train_data_size=100, test_data_size=100, wight_size=4)

    # 线性函数拟合(⽋拟合
    # test_polynomial_regression(train_data_size=100, test_data_size=100, wight_size=2)

    # ⾼阶多项式函数拟合(过拟合)
    # test_polynomial_regression(train_data_size=20, test_data_size=100, wight_size=10, num_epochs=1500)

    # ⾼阶多项式函数拟合(过拟合)，添加L2范数
    test_polynomial_regression(train_data_size=20, test_data_size=100, wight_size=10, l2_lambda=1e-2, num_epochs=2000)
