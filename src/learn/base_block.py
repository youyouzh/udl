"""
神经网络自定义块等
"""
import torch
from torch import nn

from base.util import try_gpu, Accumulator


# 自定义层
class MultilayerPerceptronBlock(nn.Module):

    def __init__(self):
        # 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        # 全连接隐藏层
        self.hidden = nn.Linear(20, 256)
        # 输出层
        self.out = nn.Linear(256, 10)

    # 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
    def forward(self, x):
        return self.out(nn.functional.relu(self.hidden(x)))


# 自定义块
class CustomSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
        for index, module in enumerate(args):
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(index)] = module

    def forward(self, x):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            x = block(x)
        return x


# 神经网络基类，声明一些常用方法和属性
class BaseNet(object):

    def __init__(self, learning_rate=0.03, num_epochs=10, batch_size=100):
        # 模型超参数
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # 尝试使用GPU
        self.device = try_gpu()

        # 数据迭代器
        self.train_data_iter = None  # 训练数据集迭代器
        self.test_data_iter = None   # 测试数据集迭代器

        # 模型结构
        self.net = None
        self.optimizer = None   # 优化算法
        self.loss_func = None  # 损失函数

        # 训练损失记录
        self.train_loss_history = None

        # 加载模型
        self.load_model()

    # 定义模型网络结构
    def define_net(self):
        pass

    # 加载模型，如果修改了超参数，需要手动调用该方法
    def load_model(self):
        pass

    def preset_hyper_param(self, learning_rate=0.03, num_epochs=5, batch_size=200):
        # 定义训练超参数
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_loss_history = torch.zeros([num_epochs])

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

    # 评估数据迭代器的损失
    def evaluate_loss(self, data_iter):
        metric = Accumulator(2)
        for X, y in data_iter:
            out = self.net(X)
            y = y.reshape(out.shape)
            loss = self.loss_func(out, y)
            metric.add(loss.sum(), loss.numel())
        return metric[0] / metric[1]

    @staticmethod
    def accuracy(y_hat, y):
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


def test_mlp():
    # 自定义层
    mlp = MultilayerPerceptronBlock()
    print('mlp output: ', mlp(torch.rand(2, 20)))

    # 参数初始化
    shared = nn.Linear(4, 4)   # 共享参数层
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), shared, nn.ReLU(), shared, nn.Linear(4, 1))
    net[0].apply(lambda m: nn.init.xavier_uniform_(m.weight))
    net[2].apply(lambda m: nn.init.constant_(m.bias, 3))

    # 直接设置参数
    net[0].weight.data[0] = 10
    net[0].weight.data[1] = 10

    # 测试模型读取
    print('state_dict: ', net[0].state_dict())
    print('net 0 weight: ', net[0].weight)   # 权重
    print('state_dict 0.weight: ', net.state_dict()['0.weight'])   # 权重
    print('state_dict 2.weight: ', net.state_dict()['2.weight'])   # 权重
    print('state_dict 4.weight: ', net.state_dict()['4.weight'])   # 权重
    print('weight grad: ', net[0].weight.grad)   # 权重梯度
    print('bias: ', net[0].bias.data)  # 偏置
    print('layer 0 all params: ', net[0].named_parameters())   # 一次乡读取所有参数

    # 保存模型
    save_file = r'../block/data/mlp.model'
    torch.save(mlp.state_dict(), save_file)
    # 加载模型
    clone = MultilayerPerceptronBlock()
    clone.load_state_dict(torch.load(save_file))


if __name__ == '__main__':
    test_mlp()
