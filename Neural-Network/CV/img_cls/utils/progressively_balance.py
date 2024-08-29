# 渐进式平衡采样，2020-ICLR-Decoupling Representation and Classifier
import sys
sys.path.append('.')
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.cifar_longtail import CifarLTDataset


class ProgressiveSampler(object):
    def __init__(self, dataset, max_epoch):
        self.max_epoch = max_epoch
        self.dataset = dataset      # dataset
        self.train_targets = [int(info[1]) for info in dataset.img_info]    #  记录各个样本的标签
        self.nums_per_cls = dataset.nums_per_cls        # 记录每个类别的样本数量

    def _cal_class_prob(self, q):
        """根据q值计算每个类的采样概率，公式中的 p_j

        Args:
            q (float): 超参数，q=1: instance balanced, q=0: clss balanced

        Returns:
            List[float]: 每个类别的采样概率
        """
        num_pow = [pow(x, q) for x in self.nums_per_cls]
        sigma_num_pow = sum(num_pow)
        cls_prob = [x / sigma_num_pow for x in num_pow]
        return cls_prob

    def _cal_pb_prob(self, t):
        """计算每个样本渐进式平衡的采样概率

        Args:
            t (int): 当前epoch数

        Returns:
            List[float]: 当前每个类别的采样概率
        """
        p_ib = self._cal_class_prob(q=1)  # -> [] * 10
        p_cb = self._cal_class_prob(q=0)
        p_pb = (1 - t/self.max_epoch) * np.array(p_ib) + (t/self.max_epoch) * np.array(p_cb)

        return p_pb.tolist()

    def __call__(self, epoch):
        """计算每个样本在当前epoch下的采样概率
        实例化pytorch的WeightedRandomSampler采样器

        Args:
            epoch (int): 当前epoch数

        Returns:
            torch.utils.data.WeightedRandomSampler: Dataloader所需要的采样器
        """
        # 计算当前每个类别的采样概率
        p_pb = self._cal_pb_prob(t=epoch)  # len=10

        # 每个样本被采样的权重，self.train_targets是所有样本的标签
        samples_weights = [p_pb[i] / self.nums_per_cls[i] for i in self.train_targets]
        samples_weights = torch.tensor(samples_weights, dtype=torch.float)
        # weights：要求是每个样本赋予weight
        # num_samples：该sampler一个epoch采样数量
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights))

        return sampler, p_pb

    def plot_line(self):
        for i in range(0, self.max_epoch, 20):
            _, weights = self(i)
            x = range(len(weights))
            plt.plot(x, weights, label="t="+str(i))
        plt.legend()
        plt.title("max epoch="+str(self.max_epoch))
        plt.xlabel("class index")
        plt.ylabel("weights")
        plt.show()


if __name__ == '__main__':
    train_dir = r"e:\data\cifar10\cifar10_train"
    from torchvision import transforms
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),  ####
        transforms.ToTensor(),
    ])
    train_data = CifarLTDataset(img_dir=train_dir, transform=transforms_train, is_train=True)

    max_epoch = 200
    sampler_generator = ProgressiveSampler(train_data, max_epoch)
    sampler_generator.plot_line()

    for epoch in range(0, max_epoch, 20):
        sampler, _ = sampler_generator(epoch)
        train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=False,
                                  num_workers=1, sampler=sampler)

        labels = []
        for data in train_loader:
            _, label, _ = data
            labels.extend(label.tolist())
        print("Epoch:{}, Counter:{}".format(epoch, Counter(labels)))