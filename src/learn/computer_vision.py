"""
计算机视觉相关实践
"""
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

from base.util import plt, show_images


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


class ResNetTrainer(object):

    @staticmethod
    # CIFAR-10数据集
    def load_cifar_10_data():
        all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
        show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

        # 对训练样本只进⾏图像增⼴，且在预测过程中不使⽤随机操作的图像增⼴
        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])
        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])


if __name__ == '__main__':
    ImageAugmentation.example()
