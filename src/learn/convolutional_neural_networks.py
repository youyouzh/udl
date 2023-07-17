"""
卷积神经网络相关
"""
import torch

from torch import nn

from base.util import Accumulator, try_gpu, Animator, Timer
from softmax_regression import load_data_fashion_mnist
from multilayer_perceptron import MultilayerPerceptron


# 自定义一个简单的卷积层
class Cov2DLayer(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    # 计算2维互相关运算
    def cov2d(self, x, k):
        h, w = k.shape
        # 初始化y
        y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = (x[i:i + h, j:j + w] * k).sum()
        return y

    def forward(self, x):
        return self.cov2d(x, self.weight) + self.bias


# LeNet神经网络，最早期的卷积神经网络之一
class LeNet(MultilayerPerceptron):

    def __init__(self):
        super().__init__()

    def define_model(self):
        pass

    def define_net(self):
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

    def ready_model(self):
        self.define_net()
        # 初始化参数
        self.net.apply(lambda m: nn.init.xavier_uniform_(m.weight) if type(m) == nn.Linear or type(m) == nn.Conv2d else None)
        # 设置在GPU上训练
        print('train device: {}'.format(self.device))
        self.net.to(self.device)
        # 设置损失函数
        self.loss_func = nn.CrossEntropyLoss()
        # 设置优化器
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)

    def evaluate_accuracy_gpu(self, device=None):
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(self.net, nn.Module):
            self.net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(self.net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = Accumulator(2)
        test_data_iter = self.get_test_data_iter()
        with torch.no_grad():
            for X, y in test_data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(self.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

    def train(self):
        """用GPU训练模型(在第六章定义)"""
        animator = Animator(xlabel='epoch', xlim=[1, self.num_epochs], legend=['train loss', 'train acc', 'test acc'])
        train_data_iter = self.get_train_data_iter()
        timer, num_batches = Timer(), len(train_data_iter)
        metric, train_loss, test_loss, train_acc, test_acc = None, None, None, None, None   # 预定义变量
        for epoch in range(self.num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(train_data_iter):
                timer.start()
                self.optimizer.zero_grad()
                # 将数据加载到GPU，加速训练
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                loss = self.loss_func(y_hat, y)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(loss * X.shape[0], self.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
                    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, ')
            test_acc = self.evaluate_accuracy_gpu()
            print('train end with epoch {}. loss: {}, tran acc: {}, test acc: {}'
                  .format(epoch + 1, train_loss, train_acc, test_acc))
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * self.num_epochs / timer.sum():.1f} examples/sec ,on {str(self.device)}')
        animator.show()


class AlexNet(LeNet):

    def define_net(self):
        self.net = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10))


class VGG(AlexNet):

    def __init__(self, conv_arch=None):
        super().__init__()
        # 该变量指定了每个VGG块⾥卷积层个数和输出通道数
        self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) if conv_arch is None else conv_arch

    def define_net(self):
        # for循环创建vgg块
        conv_blocks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in self.conv_arch:
            conv_blocks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.net = nn.Sequential(
            *conv_blocks, nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))

    # 定义VGG块
    @staticmethod
    def vgg_block(num_convs, in_channels, out_channels):
        """
        VGG块
        :param num_convs: 卷积层的数量
        :param in_channels: 输⼊通道的数量
        :param out_channels: 输出通道的数量
        :return: Sequential块
        """
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


class NiN(AlexNet):

    def define_net(self):
        self.net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            self.nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            nn.Flatten())

    @staticmethod
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


# GoogLeNet⼀共使⽤9个Inception块
class Inception(nn.Module):

    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = nn.functional.relu(self.p1_1(x))
        p2 = nn.functional.relu(self.p2_2(nn.functional.relu(self.p2_1(x))))
        p3 = nn.functional.relu(self.p3_2(nn.functional.relu(self.p3_1(x))))
        p4 = nn.functional.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(VGG):

    def define_net(self):
        # 第⼀个模块使⽤64个通道、7 × 7卷积层
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 第⼆个模块使⽤两个卷积层：第⼀个卷积层是64个通道、1 × 1卷积层；第⼆个卷积层使⽤将通道数量增加三倍的3 × 3卷积层。。这对应于Inception块中的第⼆条路径。
        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 第三个模块串联两个完整的Inception块
        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 第四模块更加复杂，它串联了5个Inception块
        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 第五模块包含输出通道数为256 + 320 + 128 + 128 = 832和384 + 384 + 128 + 128 = 1024的两个Inception块
        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Flatten())

        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


# 自定义批量正则化层
class BatchNormLayer(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = self.batch_norm(X, self.gamma, self.beta, self.moving_mean,
                                                               self.moving_var, eps=1e-5, momentum=0.9)
        return Y

    @staticmethod
    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
        if not torch.is_grad_enabled():
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
                # 这里我们需要保持X的形状以便后面可以做广播运算
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            # 训练模式下，用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 缩放和移位
        return Y, moving_mean.data, moving_var.data


# 带批量规范化的LeNet网络
class BatchNormLeNet(LeNet):

    def define_net(self):
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
            nn.Linear(84, 10))


# 残差网络块
class ResidualBlock(nn.Module):

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 有2个有相同输出通道数的3 × 3卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            # 如果想改变通道数，就需要引⼊⼀个额外的1 × 1卷积层来将输⼊变换成需要的形状后再做相加运算
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 每个卷积层后接⼀个批量规范化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        Y = nn.functional.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return nn.functional.relu(Y)


class ResNet(GoogLeNet):

    def define_net(self):
        # ResNet的前两层跟之前介绍的GoogLeNet中的⼀样，在输出通道数为64、步幅为2的7 × 7卷积层后，接步幅为2的3 × 3的最⼤汇聚层
        # 不同之处在于ResNet每个卷积层后增加了批量规范化层
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # GoogLeNet在后⾯接了4个由Inception块组成的模块。ResNet则使⽤4个由残差块组成的模块，每个模块使⽤若⼲个同样输出通道数的残差块
        # 接着在ResNet加⼊所有残差块，这⾥每个模块使⽤2个残差块
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        # 最后，与GoogLeNet⼀样，在ResNet中加⼊全局平均汇聚层，以及全连接层输出
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, 10))

    @staticmethod
    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        # 第⼀个模块的通道数同输⼊通道数⼀致。由于之前已经使⽤了步幅为2的最⼤汇聚层，所以⽆须减⼩⾼和宽。
        # 之后的每个模块在第⼀个残差块⾥将上⼀个模块的通道数翻倍，并将⾼和宽减半。
        blocks = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blocks.append(ResidualBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blocks.append(ResidualBlock(num_channels, num_channels))
        return blocks


# 稠密块
class DenseBlock(nn.Module):

    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            # 连接通道维度上每个块的输入和输出
            x = torch.cat((x, y), dim=1)
        return x

    @staticmethod
    def conv_block(input_channels, num_channels):
        # DenseNet使⽤了ResNet改良版的“批量规范化、激活和卷积”架构
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

    @staticmethod
    def transition_block(input_channels, num_channels):
        # 由于每个稠密块都会带来通道数的增加，使⽤过多则会过于复杂化模型。⽽过渡层可以⽤来控制模型复杂度
        # 它通过1 × 1卷积层来减⼩通道数，并使⽤步幅为2的平均汇聚层减半⾼和宽，从⽽进⼀步降低模型复杂度
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(ResNet):

    def define_net(self):
        # DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 接下来，类似于ResNet使用的4个残差块，DenseNet使用的是4个稠密块
        # num_channels为当前的通道数
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blocks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blocks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blocks.append(DenseBlock.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        # 与ResNet类似，最后接上全局汇聚层和全连接层来输出结果
        self.net = nn.Sequential(
            b1, *blocks,
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 10))


# 测试自动训练卷积核
def test_train_cov2d():
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    x = torch.ones((6, 8))
    # 中间四列为⿊⾊（0），其余像素为⽩⾊（1）
    x[:, 2:6] = 0
    y = torch.zeros((6, 7))
    y[:, 1:2] = 1
    y[:, 5:6] = -1
    print('x: ', x, 'y: ', y)

    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    x = x.reshape((1, 1, 6, 8))
    y = y.reshape((1, 1, 6, 7))
    lr = 3e-2  # 学习率

    for i in range(10):
        y_hat = conv2d(x)
        loss = (y_hat - y) ** 2
        conv2d.zero_grad()
        loss.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {loss.sum():.3f}')
    print('cov2d weight: ', conv2d.weight)


def test_cov2d():
    # 测试边缘检测
    x = torch.ones((6, 8))
    # 中间四列为⿊⾊（0），其余像素为⽩⾊（1）
    x[:, 2:6] = 0
    print(x)

    # 们构造⼀个⾼度为1、宽度为2的卷积核K。当进⾏互相关运算时，如果⽔平相邻的两元素相同，则输出为零，否则输出为⾮零
    k = torch.tensor([[1.0, -1.0]])
    cov2d_layer = Cov2DLayer(k.shape)
    print(cov2d_layer.cov2d(x, k))
    # 们将输⼊的⼆维图像转置，再进⾏如上的互相关运算。其输出如下，之前检测到的垂直边缘消失了
    print(cov2d_layer.cov2d(x.t(), k))


def test_cnn_net(cnn_net, resize=244, output_layer=True):
    # 定义模型
    cnn_net.ready_model()

    # 输出网络结构
    if output_layer:
        x = torch.rand(size=(1, 1, resize, resize), dtype=torch.float32, device=cnn_net.device)
        for layer in cnn_net.net:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape: \t', x.shape)
    train_data_iter, test_data_iter = load_data_fashion_mnist(cnn_net.batch_size, resize=resize)
    cnn_net.set_data_iter(train_data_iter, test_data_iter)

    # 开始训练
    cnn_net.train()


def test_le_net():
    cnn_net = LeNet()
    # 设置超参数和数据集合
    cnn_net.preset_hyper_param(batch_size=256, learning_rate=0.9, num_epochs=10)
    test_cnn_net(cnn_net, resize=28)


def test_alex_net():
    cnn_net = AlexNet()
    # 设置超参数和数据集合
    cnn_net.preset_hyper_param(batch_size=128, learning_rate=0.01, num_epochs=10)
    test_cnn_net(cnn_net)


def test_vgg():
    # 由于VGG-11⽐AlexNet计算量更⼤，因此我们构建了⼀个通道数较少的⽹络
    ratio = 4
    cnn_net = VGG()
    cnn_net.conv_arch = [(pair[0], pair[1] // ratio) for pair in cnn_net.conv_arch]
    cnn_net.preset_hyper_param(batch_size=128, learning_rate=0.05, num_epochs=10)
    test_cnn_net(cnn_net)


def test_nin():
    cnn_net = NiN()
    cnn_net.preset_hyper_param(batch_size=128, learning_rate=0.1, num_epochs=10)
    test_cnn_net(cnn_net)


def test_google_net():
    cnn_net = GoogLeNet()
    cnn_net.preset_hyper_param(batch_size=128, learning_rate=0.1, num_epochs=10)
    test_cnn_net(cnn_net, resize=96)


def test_batch_normal_le_net():
    cnn_net = BatchNormLeNet()
    # 添加正则化后，相比LeNet学习率⼤得多
    cnn_net.preset_hyper_param(batch_size=256, learning_rate=10, num_epochs=10)
    test_cnn_net(cnn_net, resize=28, output_layer=False)


def test_res_net():
    cnn_net = ResNet()
    # 添加正则化后，相比LeNet学习率⼤得多
    cnn_net.preset_hyper_param(batch_size=256, learning_rate=0.05, num_epochs=10)
    test_cnn_net(cnn_net, resize=96, output_layer=False)


def test_dense_net():
    cnn_net = DenseNet()
    # 添加正则化后，相比LeNet学习率⼤得多
    cnn_net.preset_hyper_param(batch_size=256, learning_rate=0.1, num_epochs=10)
    test_cnn_net(cnn_net, resize=96, output_layer=False)


if __name__ == '__main__':
    # test_cov2d()
    # test_train_cov2d()
    # test_le_net()
    # test_alex_net()
    # test_vgg()
    # test_nin()
    # test_google_net()
    # test_batch_normal_le_net()
    # test_res_net()
    test_dense_net()
