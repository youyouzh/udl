# MLP多层感知机multilayer perceptron
import torch
from torch import nn

from base.util import plot
from softmax_regression import SimpleSoftmaxLinearRegression, load_data_fashion_mnist


# 一些激活函数的检查测试
def activation_function():
    # RELU函数
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    legends = ['relu(x)']
    fs = [torch.relu(x).detach()]

    # sigmoid函数
    legends.append('sigmoid(x)')
    fs.append(torch.sigmoid(x).detach())

    # sigmoid函数的导数
    y = torch.relu(x)
    y.backward(torch.ones_like(x), retain_graph=True)
    legends.append('grad of sigmoid')
    fs.append(x.grad)

    # tanh函数
    legends.append('tanh(x)')
    fs.append(torch.tanh(x).detach())

    # tanh函数导数
    x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    legends.append('rad of tanh')
    fs.append(x.grad)

    # 显示图像
    plot(x.detach(), fs, 'x', legend=legends, figsize=(5, 2.5)).show()


class MultilayerPerceptron(SimpleSoftmaxLinearRegression):

    def __init__(self):
        num_inputs, num_outputs, num_hiddens = 784, 10, 256
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        self.params = [self.W1, self.b1, self.W2, self.b2]
        super().__init__()

    def define_model(self):
        def net(x):
            x = x.reshape((-1, self.W1.shape[0]))
            # 这里“@”代表矩阵乘法
            h = torch.relu(x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
        self.net = net
        # 定义损失函数
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.optimizer = torch.optim.SGD(self.params, lr=self.learning_rate)


class ConciseMultilayerPerceptron(MultilayerPerceptron):

    def define_model(self):
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(self.W1.shape[0], self.W1.shape[1]),
                                 nn.ReLU(),
                                 nn.Linear(self.W1.shape[1], self.W2.shape[1]))
        # 定义损失函数
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        # 设置初始化参数
        self.net.apply(lambda m: nn.init.normal_(m.weight, std=0.01) if type(m) == nn.Linear else m)


def test_multilayer_perceptron():
    # softmax_regression = MultilayerPerceptron()
    softmax_regression = ConciseMultilayerPerceptron()
    # 设置超参数
    softmax_regression.preset_hyper_param(num_epochs=10, learning_rate=0.1, batch_size=256)
    # 加载数据集迭代器
    train_data_iter, test_data_iter = load_data_fashion_mnist(softmax_regression.batch_size)
    # 设置数据迭代器
    softmax_regression.set_data_iter(train_data_iter, test_data_iter)
    # 开始训练
    softmax_regression.train()
    # 评估训练结果
    softmax_regression.evaluate_accuracy()
    # 模型预测
    softmax_regression.predict(test_data_iter)


if __name__ == '__main__':
    # activation_function()
    test_multilayer_perceptron()
