"""
神经网络自定义块等
"""
import torch
from torch import nn


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
