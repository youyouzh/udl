# PyTorch模型保存、加载和预训练

## PyTorch模型加载

### 保存和加载整个模型和参数

这种方式会保存整个模型的结构以及参数，会占用较大的磁盘空间， 通常不采用这种方式。

```python
torch.save(model, 'model.pkl')  # 保存
model = torch.load('model.pkl') # 加载
```

### 保存和加载模型的参数 

优点是速度快，占用的磁盘空间少， 是最常用的模型保存方法。strict参数，该参数默认是True， 表示预训练模型的网络结构与自定义的网络结构严格相同（包括名字和维度）。 如果自定义网络和预训练网络不严格相同时， 需要将不属于自定义网络的key去掉。

```python
torch.save(model.state_dict(), 'model_state_dict.pkl')
model = model.load_state_dict(torch.load(model_state_dict.pkl))
```

### 保存优化器参数

在实际场景中， 我们往往需要保存更多的信息，如优化器的参数， 那么可以通过字典的方式进行存储。

```python
# 保存
torch.save({'epoch': epochId,
            'state_dict': model.state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()}, 
             checkpoint_path + "/m-" + timestamp + str("%.4f" % best_acc) + ".pth.tar")

# 加载
def load_model(model, checkpoint, optimizer):
    model_CKPT = torch.load(checkpoint)
    model.load_state_dict(model_CKPT['state_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer
```

### 加载部分预训练模型

如果我们修改了网络， 那么就需要将这部分参数过滤掉。当两个网络的结构相同， 但是结构的命名不同时， 直接加载会报错。因此需要修改结构的key值。

```python
def load_model(model, chinkpoint, optimizer):
    model_CKPT = torch.load(checkpoint)
    model_dict = model.state_dict()
    pretrained_dict = model_CKPT['state_dict']
    # 将不在model中的参数过滤掉
    new_dict = {k, v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    
    # 加载优化器参数
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer
```

### 冻结网络的部分参数， 训练另一部分参数

当输入给模型的数据集形式相似或者相同时，常见的是利用现有的经典模型（如Residual Network、 GoogleNet等）作为backbone来提取特征，那么这些经典模型已经训练好的模型参数可以直接拿过来使用。通常情况下， 我们希望将这些经典网络模型的参数固定下来， 不进行训练，只训练后面我们添加的和具体任务相关的网络参数。

需要在网络模型中需要固定的参数位置之前添加下列代码：

```python
for p in model.parameters():
    p.requires_grad = False

# 以ResNet网络为例
# 当我们加载ResNet预训练模型之后，在ResNet的基础上连接了新的网络模块， ResNet那部分网络参数先冻结不更新
# 只更新新引入网络结构的参数
class Net(torch.nn.Module):
      def __init__(self, model, pretrained):
          super(Net, self).__init__()
          self.resnet = model(pretained)
          for p in self.parameters():
              p.requires_grad = False
          self.conv1 = torch.nn.Conv2d(2048, 1024, 1)
          self.conv2 = torch.nn.Conv2d(1024, 1024, 1)
```

必须同时在优化器中将这些参数过滤掉， 否则会报错。因为optimizer里面的参数要求required_grad为Ture

## ResNet预训练的常规操作

Pytorch的torchvision.models中提供了一些经典的网络模型及其预训练的参数：alexnet, resnet, vgg, squeezenet, inception等。

```python
# 加载网络模型及其训练的参数，pretrained=False只加载网络结构， 不加载训练的参数
import torchvision.models as models
resnet34 = models.resnet34(pretrained=True) # 默认为False

#  resnet网络的最后一层对应1000个类别， 如果我们自己的数据只有10个类别， 那么可以进行如下修改
model = models.resnet50(pretrained=True)
fc_inDim = model.fc.in_features
# 修改为10个类别
model.fc = torch.nn.Linear(fc_inDim, 10)
```

增加新的网络层：如果需要在resnet的基础上增加自己的网络层，需要自己先定义一个类似的网络， 然后再把预训练的模型参数加载进来。

