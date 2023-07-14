# 多项式回归
import math
import numpy as np
import torch
from torch import nn

from base.util import plt, plot, Accumulator, Animator, sgd
from linear.linear_regression import ManualLinearRegression
from linear.multilayer_perceptron import MultilayerPerceptron


# 生成多项式数据集
def generate_polynomial_data():
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
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
    return true_w, features, poly_features, labels


def test_generate_polynomial_data():
    true_w, features, poly_features, labels = generate_polynomial_data()
    # 显示图像
    plt.scatter(features, labels, 1)
    plt.show()


class PolynomialRegression(MultilayerPerceptron):

    def __init__(self):
        super().__init__()

    def define_model(self):
        self.net = nn.Sequential(nn.Linear(self.w.shape[0], 1, bias=False))
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.w, lr=self.learning_rate)




if __name__ == '__main__':
    test_generate_polynomial_data()



