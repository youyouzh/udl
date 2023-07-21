"""
通用的训练器，定义一些比较常用的训练相关函数，比如准确率评估等
"""
import torch
from torch import nn

from base.util import try_gpu, try_all_gpus, Accumulator


class BaseTrainer(object):
    device = try_gpu()
    devices = try_all_gpus()

    @staticmethod
    def accuracy(y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            # 取概率最大的类别作为预测结果
            y_hat = y_hat.argmax(axis=1)
        # 类型敏感，先检查类型
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    @staticmethod
    def evaluate_accuracy(net, test_data_iter):
        """计算在测试数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in test_data_iter:
                # 遍历测试数据集，计算模型预测结果和世界结果差距并评估准确率
                metric.add(BaseTrainer.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    @staticmethod
    def evaluate_accuracy_gpu(net, data_iter, device=None):
        """Compute the accuracy for a model on a dataset using a GPU."""
        if isinstance(net, nn.Module):
            net.eval()  # Set the model to evaluation mode
            if not device:
                device = next(iter(net.parameters())).device
        # No. of correct predictions, no. of predictions
        metric = Accumulator(2)

        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # Required for BERT Fine-tuning (to be covered later)
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(BaseTrainer.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]
