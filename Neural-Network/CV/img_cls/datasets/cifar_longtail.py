import os
import random
from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cls_num = len(names)

    def __init__(self, img_dir, transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_info = []  # path, label ...
        self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        img_path, label_id = self.img_info[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label_id, img_path

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.img_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, _ in os.walk(self.img_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                # img_names = [x for x in img_names if x.endswith('.png')]
                # 遍历图片
                for img_name in img_names:
                    img_path = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    label = int(sub_dir)
                    self.img_info.append((img_path, int(label)))
        random.shuffle(self.img_info)   # 将数据顺序打乱


class CifarLTDataset(CifarDataset):
    """长尾分布的cifar10数据集

    Args:
        img_dir (str): 图片路径
        transform (_type_, optional): _description_. Defaults to None.
        imb_factor (float, optional): 取值(0, 1]，长尾分布下降因子， 值越小越陡峭. Defaults to 0.01.
        is_train (bool, optional): 是否为训练数据，不是的话分布为原始分布. Defaults to True.
    """
    def __init__(self, img_dir, transform=None, imb_factor=0.01, is_train=True):
        super(CifarLTDataset, self).__init__(img_dir, transform=transform)
        self.imb_factor = imb_factor
        self.cache_file = os.path.dirname(self.img_dir) + f'/cifar10_longtail{self.imb_factor}.txt'
        if is_train:
            if os.path.exists(self.cache_file):
                print(f'read cache from {self.cache_file}')
                self.img_info, self.nums_per_cls = self._read_cache()
            else:
                self.img_info, self.nums_per_cls = self._select_img()      # 采样获得符合长尾分布的数据量
        else:
            # 非训练状态，采用均衡数据集测试
            self.nums_per_cls = []
            label_list = [label for p, label in self.img_info]  # 获取每个标签
            for n in range(self.cls_num):
                self.nums_per_cls.append(label_list.count(n))   # 统计每个类别数量

    def _read_cache(self):
        """读取存放长尾分布样本信息的缓存文件

        """
        new_lst = []
        with open(self.cache_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                p, label = line.strip().split()
                new_lst.append((p, int(label)))

        nums_per_cls = []
        label_list = [label for p, label in new_lst]  # 获取每个标签
        for n in range(self.cls_num):
            nums_per_cls.append(label_list.count(n))  # 统计每个类别数量
        return new_lst, nums_per_cls

    def _select_img(self):
        """
        step1: 根据下降因子获得每个类的数量
        step2: 根据数量随机抽取每个类的样本
        """
        nums_per_cls = self._get_img_num_per_cls()     # 计算每个类的样本数
        
        new_lst = []
        for n, img_num in enumerate(nums_per_cls):
            lst_tmp = [info for info in self.img_info if info[1] == n]  # 获取第n类别数据信息
            lst_tmp = random.sample(lst_tmp, img_num)
            new_lst.extend(lst_tmp)
        random.shuffle(new_lst)
        
        # 保存到文件中，方便复现
        with open(self.cache_file, 'w', encoding='utf-8') as outfile:
            [outfile.write(f'{p} {label}\n') for p, label in new_lst]

        return new_lst, nums_per_cls

    def _get_img_num_per_cls(self):
        """
        依长尾分布计算每个类别应有多少张样本
        """
        img_max = len(self.img_info) / self.cls_num  # cifar: 5000
        img_num_per_cls = []
        for cls_idx in range(self.cls_num):
             # 指数分布 数字越小越陡峭
            num = img_max * (self.imb_factor ** (cls_idx / (self.cls_num - 1.0)))  # cls_idx [0-9]
            img_num_per_cls.append(int(num))
        return img_num_per_cls


if __name__ == "__main__":

    root_dir = r"E:\data\cifar10\cifar10_train"
    train_dataset = CifarLTDataset(root_dir, imb_factor=0.01)
    print('new dataset size:', len(train_dataset))
    print(train_dataset.nums_per_cls)

    y = train_dataset.nums_per_cls
    x = range(len(y))
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()