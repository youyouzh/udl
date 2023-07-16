import math
import random
import numpy as np
import torch
from torch import nn

from base.util import plot, plt, sgd, torch_data_iter, try_gpu


# 计算正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


def normal_plot():
    # 再次使⽤numpy进⾏可视化
    x = np.arange(-7, 7, 0.01)
    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params]).show()


def synthetic_data(w, b, sample_size=1000, mu=0, sigma=0.01):
    """⽣成y=Xw+b+噪声"""
    x = torch.normal(0, 1, (sample_size, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(mu, sigma, y.shape)
    return x, y.reshape((-1, 1))


def test_synthetic_data():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b)
    print('features:', features[0], '\nlabel:', labels[0])
    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    plt.show()


def manual_data_iter(features, labels, batch_size):
    """
    手动实现的数据迭代器，⽣成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
    :param features 特征矩阵
    :param labels 标签向量
    :param batch_size 批量⼤⼩
    """
    # 生成随机下标
    sample_size = len(features)
    indices = list(range(sample_size))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, sample_size, batch_size):
        # 截取batch_size的样本数据
        batch_indices = torch.tensor(indices[i: min(i + batch_size, sample_size)])
        yield features[batch_indices], labels[batch_indices]


# 手动实现的线性分类器
class ManualLinearRegression(object):

    def __init__(self):
        # 待训练参数
        self.w = None
        self.b = None

        # 模型结构
        self.net = None
        self.optimizer = None   # 优化算法
        self.loss_func = None  # 损失函数

        # 数据迭代器
        self.train_data_iter = None  # 训练数据集迭代器
        self.test_data_iter = None   # 测试数据集迭代器

        # 全量输入特征和标签
        self.features = None
        self.labels = None

        # 保存训练过程
        self.train_loss_history = None

        # 模型超参数
        self.learning_rate = 0.03
        self.num_epochs = 3
        self.batch_size = 100

        # 尝试使用GPU
        self.device = try_gpu()

        # 定义模型结构
        self.define_model()

    def define_model(self):
        """
        初始化模型参数
        定义模型，设定模型结构，损失函数，优化方法
        """
        # 均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
        self.w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

        # 线性回归模型
        self.net = lambda x: torch.matmul(x, self.w) + self.b
        self.loss_func = lambda y_hat, y: (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        self.optimizer = sgd

    def preset_hyper_param(self, learning_rate=0.03, num_epochs=5, batch_size=200):
        # 定义训练超参数
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_loss_history = torch.zeros([num_epochs])

    # 设置全量的输入特征和标签
    def set_features_labels(self, features, labels):
        """
        设置训练全量数据，包括特征和标签，会同时设置训练数据迭代器和测试数据迭代器
        :param features: 特征
        :param labels: 标签
        :return:
        """
        self.features = features
        self.labels = labels
        self.train_data_iter = manual_data_iter

    def set_data_iter(self, train_iter, test_iter=None):
        """
        设置数据迭代器
        :param train_iter: 训练数据集迭代器
        :param test_iter: 测试数据集迭代器
        """
        self.train_data_iter = train_iter
        self.test_data_iter = test_iter

    def get_train_data_iter(self):
        """
        获取数据迭代器，返回训练数据集迭代器data_iter
        """
        if not self.train_data_iter:
            raise Exception('请先设置训练数据迭代器')
        return self.train_data_iter

    def get_test_data_iter(self):
        if not self.test_data_iter:
            raise Exception('请先设置测试数据迭代器')
        return self.test_data_iter

    def get_features_labels(self):
        if self.features is None:
            raise Exception("请先设置全量输入特征和标签")
        return self.features, self.labels

    def train(self):
        # 迭代训练
        features, labels = self.get_features_labels()
        for epoch in range(self.num_epochs):
            for X, y in self.get_train_data_iter()(self.features, self.labels, self.batch_size):
                loss = self.loss_func(self.net(X), y)  # X和y的小批量损失
                # 因为loss形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
                # 并以此计算关于[w,b]的梯度
                loss.sum().backward()
                self.optimizer([self.w, self.b], self.learning_rate, self.batch_size)  # 使用参数的梯度更新参数
            with torch.no_grad():
                # 全量数据数据的损失
                total_loss = self.loss_func(self.net(features), labels).mean()
                print(f'epoch {epoch + 1}, loss {float(total_loss):f}')
                self.train_loss_history[epoch] = total_loss

    def evaluate(self, true_w, true_b):
        print(f'true_w: {true_w}, w: {self.w}, 估计误差：{true_w - self.w.reshape(true_w.shape)}')
        print(f'true_b: {true_b}, b: {self.b}, 估计误差：{true_b - self.b}')
        plot(torch.arange(self.num_epochs), self.train_loss_history,
             xlabel='epoch', ylabel='loss', figsize=(4.5, 2.5)).show()


# 使用torch库简化的线性分类器
class SimpleLinearRegression(ManualLinearRegression):

    def __init__(self):
        super().__init__()

    def define_model(self):
        # 定义一个输入是2维，输出是1维的神经网络层
        self.net = nn.Sequential(nn.Linear(2, 1))
        # 计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数
        self.loss_func = nn.MSELoss()
        # 设置小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        # 初始化模型参数
        self.net[0].weight.data.normal_(0, 0.01)
        self.net[0].bias.data.fill_(0)

    def set_features_labels(self, features, labels):
        self.features = features
        self.labels = labels

        # 数据迭代器，实现小批量样本抽取
        self.train_data_iter = torch_data_iter(features, labels, self.batch_size)

    def train(self):
        train_data_iter = self.get_train_data_iter()
        features, labels = self.get_features_labels()
        for epoch in range(self.num_epochs):
            for X, y in train_data_iter:
                # 生成预测并计算损失`loss`
                loss = self.loss_func(self.net(X), y)
                self.optimizer.zero_grad()
                # 通过进行反向传播来计算梯度
                loss.backward()
                # 通过调用优化器来更新模型参数
                self.optimizer.step()
            # 全量数据数据的损失
            total_loss = self.loss_func(self.net(features), labels)
            print(f'epoch {epoch + 1}, loss {total_loss:f}')
            self.train_loss_history[epoch] = total_loss.detach()
        self.w = self.net[0].weight.data
        self.b = self.net[0].bias.data


def test_manual_linear():
    # linear_regression = ManualLinearRegression()
    linear_regression = SimpleLinearRegression()
    # 定义训练超参数
    linear_regression.preset_hyper_param(num_epochs=50)
    # 定义数据集
    preset_w = torch.tensor([2, -3.4])
    preset_b = 4.2
    features, labels = synthetic_data(preset_w, preset_b)
    linear_regression.set_features_labels(features, labels)
    # 训练模型
    linear_regression.train()
    # 评估模型结果
    linear_regression.evaluate(preset_w, preset_b)


if __name__ == '__main__':
    test_manual_linear()
