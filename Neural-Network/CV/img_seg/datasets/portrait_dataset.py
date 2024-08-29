import os

import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import tqdm


class PortraitDataset2000(Dataset):
    """
    Deep Automatic Portrait Matting  2000数据集读取
    """
    names = ("bg", "portrait")
    cls_num = 2

    def __init__(self, root_dir, transform=None, is_train=True):
        super(PortraitDataset2000, self).__init__()
        self.img_dir = os.path.join(root_dir, 'training' if is_train else 'testing')
        self.transform = transform
        self.label_path_list = list()
        # 获取mask的path
        self._get_img_path()

    def __getitem__(self, index):
        # step1：读取样本，得到ndarray形式
        path_label = self.label_path_list[index]
        path_img = path_label[:-10] + ".png"
        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_gray = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)  # 1个通道值是一样的 # 0-255，边界处是平滑的  (800, 600, 1)

        # step2: 图像预处理
        if self.transform:
            # albumentations
            transformed = self.transform(image=img_rgb, mask=msk_gray)
            img_rgb = transformed['image']
            msk_gray = transformed['mask']   # transformed后仍旧是连续的

        # step3：处理数据成为模型需要的形式
        img_rgb = img_rgb.transpose((2, 0, 1))      # hwc --> chw
        img_chw_tensor = torch.from_numpy(img_rgb).float()

        msk_gray = msk_gray / 255.    # [0,255] scale [0,1] 连续变量
        label_tensor = torch.tensor(msk_gray, dtype=torch.float).unsqueeze(0)  # 标签输出为 0-1之间的连续变量 ，shape=(1, 512, 512)
        # label_tensor = torch.from_numpy(msk_gray).float().unsqueeze(0)

        return img_chw_tensor, label_tensor  # (3, 512, 512)  (1, 512, 512)

    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):
        file_list = os.listdir(self.img_dir)
        file_list = list(filter(lambda x: x.endswith("_matte.png"), file_list))
        path_list = [os.path.join(self.img_dir, name) for name in file_list]
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception("\nroot_dir:{} is a empty dir! Please checkout your path to images!".format(self.img_dir))
        self.label_path_list = path_list


class PortraitDataset34427(Dataset):
    """
    Deep Automatic Portrait Matting  34427，数据集读取
    ├─clip_img
    │  └─1803151818
    │      └─clip_00000000
    └─matting
        └─1803151818
            └─matting_00000000
    """
    names = ("bg", "portrait")
    cls_num = 2

    def __init__(self, root_dir, transform=None, ext_num=None):
        super(PortraitDataset34427, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.ext_num = ext_num

        # 获取mask的path
        self._get_img_path()

        # 截取部分数据
        if self.ext_num:
            self.img_info = self.img_info[:self.ext_num]

    def __getitem__(self, index):

        path_img, path_label = self.img_info[index]

        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_bgra = cv2.imread(path_label, cv2.IMREAD_UNCHANGED)  # 4通道图像  (800, 600, 4)
        msk_gray = msk_bgra[:, :, 3]  # 提取透明度 [0-255]

        if self.transform:
            transformed = self.transform(image=img_rgb, mask=msk_gray)
            img_rgb = transformed['image']
            msk_gray = transformed['mask']

        img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
        msk_gray = msk_gray/255.  # [0-1]连续变量

        label_out = torch.tensor(msk_gray, dtype=torch.float)
        img_chw_tensor = torch.from_numpy(img_rgb).float().unsqueeze(0)

        return img_chw_tensor, label_out

    def __len__(self):
        return len(self.img_info)

    def _get_img_path(self):
        img_dir = os.path.join(self.root_dir, "clip_img")
        img_lst, msk_lst = [], []

        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if not file.endswith(".jpg"):
                    continue
                if file.startswith("._"):
                    continue
                path_img = os.path.join(root, file)
                img_lst.append(path_img)

        for path_img in img_lst:
            path_msk = path_img.replace("clip_img", "matting"
                                        ).replace("clip_0", "matting_0"
                                                  ).replace(".jpg", ".png")
            if os.path.exists(path_msk):
                msk_lst.append(path_msk)
            else:
                print("path not found: {}\n path_img is: {}".format(path_msk, path_img))

        if len(img_lst) != len(msk_lst):
            raise Exception("\nimg info Error, img can't match with mask. got {} img, but got {} mask".format(
                len(img_lst), len(msk_lst)))
        if len(img_lst) == 0:
            raise Exception("\nroot_dir:{} is a empty dir! Please checkout your path to images!".format(self.root_dir))

        self.img_info = [(i, m) for i, m in zip(img_lst, msk_lst)]
        random.shuffle(self.img_info)


if __name__ == "__main__":

    data_dir1 = r"E:\data\Portrait-dataset-2000"
    data_dir2 = r"E:\data\Matting_Human_Half"

    import albumentations as A
    transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset1 = PortraitDataset2000(data_dir1, transform)
    dataset2 = PortraitDataset34427(data_dir2, transform)
    
    from torch.utils.data import ConcatDataset
    all_set = ConcatDataset([dataset1, dataset2])
    img, label = all_set[2500]
    print(len(all_set))
    
    print(img.shape, label.shape)