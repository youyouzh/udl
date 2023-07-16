import os.path

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

from base.util import plt, plot, Accumulator, Animator, sgd
from linear.linear_regression import ManualLinearRegression

DATA_LOADER_WORKERS = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use device: {}'.format(device))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


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


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    with_download = False
    if not os.path.isdir( r'data\FashionMNIST\t10k-images-idx3-ubyte.gz'):
        # 如果没有下载文件则设置需要下载
        with_download = True
    mnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=with_download)
    mnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=with_download)
    # CPU 不能带参数 num_workers=DATA_LOADER_WORKERS
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


def test_load_data_fashion_mnist():
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        show_images(X.reshape(32, 64, 64), 4, 8, titles=get_fashion_mnist_labels(y))
        break


def softmax(x):
    # 对每个项求幂（使⽤exp）
    x_exp = torch.exp(x)
    # 对每⼀⾏求和（⼩批量中每个样本是⼀⾏），得到每个样本的规范化常数
    partition = x_exp.sum(1, keepdim=True)
    # 将每⼀⾏除以其规范化常数，确保结果的和为1
    return x_exp / partition   # 这里应用了广播机制


class SoftmaxLinearRegression(ManualLinearRegression):

    def define_model(self):
        num_inputs = 784
        num_outputs = 10
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

        # softmax线性回归模型
        # 将数据传递到模型之前，我们使⽤reshape函数将每张原始图像展平为向量
        self.net = lambda x: softmax(torch.matmul(x.reshape((-1, self.w.shape[0])), self.w) + self.b)
        # 交叉熵损失函数
        self.loss_func = lambda y_hat, y: - torch.log(y_hat[range(len(y_hat)), y])
        # 定义优化函数
        self.optimizer = sgd

    def train_epoch(self, data_iter):
        """训练模型一个迭代周期"""
        # 将模型设置为训练模式
        if isinstance(self.net, torch.nn.Module):
            self.net.train()
        # 训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in data_iter:
            # 计算梯度并更新参数
            y_hat = self.net(X)
            loss = self.calc_loss(y_hat, y)
            if isinstance(self.optimizer, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            else:
                # 使用定制的优化器和损失函数
                loss.sum().backward()
                self.optimizer([self.w, self.b], self.learning_rate, self.batch_size)
            metric.add(float(loss.sum()), self.accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    def calc_loss(self, y_hat, y):
        # 计算损失
        return self.loss_func(y_hat, y)

    def train(self):
        """训练模型（定义见第3章）"""
        print('----> begin train. epoch size: {}'.format(self.num_epochs))
        animator = Animator(xlabel='epoch', xlim=[1, self.num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        train_data_iter = self.get_train_data_iter()
        train_loss, train_acc, test_acc = 0.0, 0.0, 0.0
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(train_data_iter)
            train_loss, train_acc = train_metrics
            test_acc = self.evaluate_accuracy()
            animator.add(epoch + 1, train_metrics + (test_acc,))
            print(f'epoch {epoch + 1}, train loss: {train_loss}, train acc: {train_acc}, test acc: {test_acc}')
        animator.show()
        # 训练损失和准确率检查
        assert train_loss < 0.5, train_loss
        assert 1 >= train_acc > 0.7, train_acc
        assert 1 >= test_acc > 0.7, test_acc

    def accuracy(self, y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            # 取概率最大的类别作为预测结果
            y_hat = y_hat.argmax(axis=1)
        # 类型敏感，先检查类型
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(self):
        """计算在测试数据集上模型的精度"""
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in self.get_test_data_iter():
                # 遍历测试数据集，计算模型预测结果和世界结果差距并评估准确率
                metric.add(self.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

    def predict(self, test_iter, n=6):
        """预测标签"""
        for X, y in test_iter:
            break
        trues = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(self.net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


class SimpleSoftmaxLinearRegression(SoftmaxLinearRegression):

    def define_model(self):
        # 在线性层前定义了展平层（flatten），来调整网络输入的形状
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        # 定义损失函数
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # 定义优化算法
        self.optimizer = torch.optim.SGD(self.net.parameters(), self.learning_rate)
        # 设置初始化参数
        self.net.apply(lambda m: nn.init.normal_(m.weight, std=0.01) if type(m) == nn.Linear else m)


def test_softmax_regression():
    # softmax_regression = SoftmaxLinearRegression()
    softmax_regression = SimpleSoftmaxLinearRegression()
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
    test_softmax_regression()
