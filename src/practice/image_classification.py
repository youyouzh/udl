"""
计算机视觉相关实践
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from PIL import Image

from base.datasets import CIFA10DataSet, HotDogDataSet
from base.trainer import BaseTrainer
from base.util import plt, show_images, Timer, Animator, Accumulator
from base import u_log as log
from learn.convolutional_neural_networks import res_net_18


class ImageAugmentation(object):

    @staticmethod
    def show_image():
        plt.rcParams['figure.figsize'] = (4.5, 3.5)
        img = Image.open(r'data/cat1.jpg')
        plt.imshow(img)
        plt.show()

    @staticmethod
    def show_augmentation_images(img, aug_fn, num_rows=2, num_cols=4, scale=1.5):
        y = [aug_fn(img) for _ in range(num_rows * num_cols)]
        show_images(y, num_rows, num_cols, scale=scale)

    @staticmethod
    def example():
        # ImageAugmentation.show_image()
        # 有50%的⼏率使图像向左或向右翻转
        image = Image.open(r'data/cat1.jpg')
        ImageAugmentation.show_augmentation_images(image, transforms.RandomHorizontalFlip())

        # 图像各有50%的⼏率向上或向下翻转
        ImageAugmentation.show_augmentation_images(image, transforms.RandomVerticalFlip())

        # 码将随机裁剪⼀个⾯积为原始⾯积10%到100%的区域，该区域的宽⾼⽐从0.5〜2之间随机取值。然后，区域的宽度和⾼度都被缩放到200像素
        shape_fug_fn = transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
        ImageAugmentation.show_augmentation_images(image, shape_fug_fn)

        # 以改变图像颜⾊的四个⽅⾯：亮度、对⽐度、饱和度和⾊调
        aug_fn = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
        ImageAugmentation.show_augmentation_images(image, aug_fn)

        # 随机更改图像的⾊调
        aug_fn = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
        ImageAugmentation.show_augmentation_images(image, aug_fn)

        # 同时随机更改图像的亮度（brightness）、对⽐度（contrast）、饱和度（saturation）和⾊调（hue）
        color_aug_fn = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ImageAugmentation.show_augmentation_images(image, color_aug_fn)

        # 过使⽤⼀个Compose实例来综合上⾯定义的不同的图像增⼴⽅法，并将它们应⽤到每个图像
        aug_fns = transforms.Compose([transforms.RandomHorizontalFlip(), color_aug_fn, shape_fug_fn])
        ImageAugmentation.show_augmentation_images(image, aug_fns)

        # 使⽤ToTensor实例将⼀批图像转换为深度学习框架所要求的格式，即形状为（批量⼤⼩，通道数，⾼度，宽度）的32位浮点数，取值范围为0〜1
        transforms.ToTensor()


class ResNetTrainer(BaseTrainer):

    @staticmethod
    def train_batch(x, y, net, loss_fn, optimizer, devices):
        """⽤多GPU进⾏⼩批量训练"""
        if isinstance(x, list):
            # 微调BERT中所需
            x = [x.to(devices[0]) for x in x]
        else:
            x = x.to(devices[0])
        y = y.to(devices[0])
        net.train()
        optimizer.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.sum().backward()
        optimizer.step()
        train_loss_sum = loss.sum()
        train_acc_sum = BaseTrainer.accuracy(pred, y)
        log.info('train batch finish. loss: {}, acc sum: {}'.format(train_loss_sum, train_acc_sum))
        return train_loss_sum, train_acc_sum

    @staticmethod
    def train(net, train_data_iter, test_data_iter, loss_fn, optimizer,
              num_epochs, devices=BaseTrainer.devices):
        """用多GPU进行模型训练"""
        timer, num_batches = Timer(), len(train_data_iter)
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        for epoch in range(num_epochs):
            # 4个维度：储存训练损失，训练准确度，实例数，特点数
            metric = Accumulator(4)
            for i, (features, labels) in enumerate(train_data_iter):
                timer.start()
                loss, acc = ResNetTrainer.train_batch(features, labels, net, loss_fn, optimizer, devices)
                metric.add(loss, acc, labels.shape[0], labels.numel())
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[2], metric[1] / metric[3],
                                  None))
            test_acc = BaseTrainer.evaluate_accuracy_gpu(net, test_data_iter)
            animator.add(epoch + 1, (None, None, test_acc))
            log.info('finish epoch train. test acc: {}'.format(test_acc))
        log.info(f'loss: {metric[0] / metric[2]:.3f}, train acc: {metric[1] / metric[3]:.3f}, test acc: {test_acc:.3f}')
        log.info(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
        animator.show()

    @staticmethod
    def train_with_data_aug(train_augs, test_augs, net, lr=0.001, batch_size=256):
        log.info('begin train with data aug')
        train_data_iter = CIFA10DataSet.load_cifar_10(True, train_augs, batch_size)
        test_test_iter = CIFA10DataSet.load_cifar_10(False, test_augs, batch_size)

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        ResNetTrainer.train(net, train_data_iter, test_test_iter, loss_fn, optimizer, num_epochs=10)
        log.info('end train with data aug')

    @staticmethod
    def example():
        batch_size, net = 256, res_net_18(10, 3)
        net.apply(lambda m: nn.init.xavier_uniform_(m.weight) if type(m) in [nn.Linear, nn.Conv2d] else None)
        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])
        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        # 使用图像增广时，train acc更小，而 test acc更大，train loss 更小，这说明图像增广减轻了过拟合
        # 把亮度色调那些变换都开，设置为0.5，测试精度直接增加了1个多点
        ResNetTrainer.train_with_data_aug(train_augs, test_augs, net, lr=0.01)


class ResNetFineTuneTrainer(ResNetTrainer):
    """
    1.使用4e-4的学习率，速度变快了（176.2->190.3 examples/sec）,准确度提升了（train acc 0.933->0.951）
    2.更改了超参数：weight decay(0.001->0.01)
    速度提升明显，泛化能力提升了，但是在训练集上的精确度下降了。
    我觉得这样子是合理的，提升L2重要性，即降低了过拟合可能性，即提升了泛化能力
    """

    @staticmethod
    def example():
        # 们使⽤在ImageNet数据集上预训练的ResNet-18作为源模型，们指定pretrained=True以⾃动下载预训练的模型参数
        finetune_net = torchvision.models.resnet18(pretrained=True)
        # 修改输出层为类别个数，并初始化，原来的输出层实1000个分类
        finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
        nn.init.xavier_uniform_(finetune_net.fc.weight)

        # 数据集
        train_augs, test_augs = ResNetFineTuneTrainer.get_aug()
        batch_size = 128
        data_dir = HotDogDataSet.download()
        train_data_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_augs),
            batch_size=batch_size, shuffle=True)
        test_test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'test'), transform=test_augs),
            batch_size=batch_size)
        ResNetFineTuneTrainer.train_fine_tuning(finetune_net, train_data_iter, test_test_iter,
                                                learning_rate=5e-5, num_epochs=5, param_group=True)

    @staticmethod
    def get_aug():
        # 从图像中裁切随机⼤⼩和随机⻓宽⽐的区域，然后将该区域缩放为224×224输⼊图像
        # 使⽤RGB通道的均值和标准差，以标准化每个通道
        normalize = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize])
        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize([256, 256]),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize])
        return train_augs, test_augs

    @staticmethod
    def train_fine_tuning(net, train_data_iter, test_data_iter, learning_rate, num_epochs=5, param_group=True):
        log.info('begin train with fine tuning.')
        loss = nn.CrossEntropyLoss(reduction="none")
        if param_group:
            # 如果param_group=True，输出层中的模型参数将使用十倍的学习率
            params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
            optimizer = torch.optim.SGD([{'params': params_1x},
                                         {'params': net.fc.parameters(),
                                        'lr': learning_rate * 10}],
                                        lr=learning_rate, weight_decay=0.001)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
        ResNetTrainer.train(net, train_data_iter, test_data_iter, loss, optimizer, num_epochs)


if __name__ == '__main__':
    # ImageAugmentation.example()
    ResNetTrainer.example()
    # ResNetFineTuneTrainer.example()
