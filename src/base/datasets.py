"""
全局数据集管理
"""
import os
import torchvision

from torch.utils import data

from base.util import try_gpu, show_images, download_extract

# 全局数据集存放路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
DATA_LOAD_WORKERS = 0  # 需要GPU才可以设置多个WORK
device = try_gpu()


class CIFA10DataSet(object):

    @staticmethod
    # CIFAR-10数据集
    def download_cifar_10_data():
        all_images = torchvision.datasets.CIFAR10(train=True, root=DATA_DIR, download=True)
        show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

    @staticmethod
    def get_aug():
        # 为了在预测过程中得到确切的结果，我们通常对训练样本只进⾏图像增⼴，且在预测过程中不使⽤随机操作的图像增⼴
        train_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])
        test_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        return train_aug, test_aug

    @staticmethod
    def load_cifar_10(is_train, augmentations, batch_size):
        dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=is_train, transform=augmentations, download=True)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=DATA_LOAD_WORKERS)
        return dataloader


class HotDogDataSet(torchvision.datasets.VisionDataset):

    @staticmethod
    def download():
        filename = 'hotdog.zip'
        return download_extract(filename)

    @staticmethod
    def get_data_iter():
        train_images = torchvision.datasets.ImageFolder(os.path.join(HotDogDataSet.download(), 'train'))
        test_images = torchvision.datasets.ImageFolder(os.path.join(HotDogDataSet.download(), 'test'))
        return train_images, test_images

    @staticmethod
    def example():
        train_images, _ = HotDogDataSet.get_data_iter()
        hotdog_images = [train_images[i][0] for i in range(8)]
        not_hotdog_images = [train_images[-i - 1][0] for i in range(8)]
        show_images(hotdog_images + not_hotdog_images, 2, 8, scale=1.4)


if __name__ == '__main__':
    HotDogDataSet.download()
