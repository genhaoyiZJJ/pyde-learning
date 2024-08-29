"""
扩增数据集：使用coco数据集作为背景，和肖像图片融合
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
import shutil
from pycocotools.coco import COCO
from functools import partial


class CocoImg(object):
    """
    根据指定类别挑选图片，并过滤掉图片中存在的类别
    如挑选室外类别，但是不希望图片中出现person
    pycocotool使用：
    https://blog.csdn.net/u013832707/article/details/94445495
    https://zhuanlan.zhihu.com/p/70878433

    args:
        coco_root: 根目录， 下属应当有images 和 annotations两个目录
        data_type(str), val2017 train2017等
        cats_in: list, 选择的超类别 eg:["outdoor"]
        cats_out: list, 过滤的子类别 eg:["person"]
        cats_in, cats_out 分别是挑选的类别和过滤类别
    """
    def __init__(self, coco_root, data_type, cats_in, cats_out):
        self.coco_root = coco_root
        self.data_type = data_type
        self.ann_path = None
        self.coco = self._load_coco()  ## COCO helper class
        self.super_cats_in = cats_in
        self.super_cats_out = cats_out
        self.cats_in_ids = None     # 全图类别id， 用于挑选图片
        self.cats_out_ids = None    # bbox目标id， 用于过滤图片
        self.coco_super_cats = []
        self.coco_cats = []
        self.img_list = []  # 所有满足条件的图像id

        # 0. 加载coco类别，获得超类和子类
        self._get_coco_cats()
        # 1. 根据类别名获取对应图片类别id
        self._get_cats_ids()  # str --> ids
        # 2. 根据类别id获取所有的img_list(图片id)
        self._get_img_list()
        # 3. 执行过滤条件，过滤剩下的图片id
        self._filter_img_list()

    def __getitem__(self, item):
        img_id = self.img_list[item]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, self.data_type, img_info['file_name'])
        return img_id, img_path

    def __len__(self):
        return len(self.img_list)

    def _load_coco(self):
        self.ann_path = os.path.join(self.coco_root, f'annotations/instances_{self.data_type}.json')
        coco = COCO(self.ann_path)
        return coco

    def _get_cats_ids(self):
        self.cats_in_ids = self.coco.getCatIds(supNms=self.super_cats_in)  # 选大图
        self.cats_out_ids = self.coco.getCatIds(supNms=self.super_cats_out)  # 过滤小目标

    def _get_img_list(self):
        # 每个类别对应的图片id
        for k, v in self.coco.catToImgs.items():
            # 如果类别是我们想要的
            if k in self.cats_in_ids:
                # 添加此类别对应的图像id
                self.img_list.extend(v)
        # [self.img_list.extend(v) for k, v in self.coco.catToImgs.items() if k in self.cats_in_ids]

    def filter_func_by_id(self, img_id, cats_out_ids):
        """
        判断单张图片里的obj是否有不希望的类别
        cats_out_ids: 不需要类别的id
        返回True or False
        """
        ann_idx = self.coco.getAnnIds(imgIds=img_id)  # 获得这张图片里的所有标注id
        anns = self.coco.loadAnns(ann_idx)  # 根据标注id获得标注信息
        img_obj_idx = []
        _ = [img_obj_idx.append(a["category_id"]) for a in anns]  # 获得标注目标的类别名称
        is_inter = bool(set(img_obj_idx).intersection(cats_out_ids))  # 判断是否和cats_out_ids有交集
        return bool(1-is_inter)

    def _filter_img_list(self):
        # 设置过滤函数：self.filter_func_by_id函数参数cats_out_ids固定为self.cats_out_ids
        # 过滤函数接受img_id作为输入
        filter_func = partial(self.filter_func_by_id, cats_out_ids=self.cats_out_ids)
        print(f"before filter, img length:{len(self.img_list)}")
        # 只保留图片列表里满足过滤函数的图片
        self.img_list = list(filter(filter_func, self.img_list))
        print(f"after filter, img length:{len(self.img_list)}")

    def _get_coco_cats(self):
        # 遍历每个类别
        for k, v in self.coco.cats.items():  # dict
            self.coco_super_cats.append(v["supercategory"])
            self.coco_cats.append(v["name"])
        self.coco_super_cats = list(set(self.coco_super_cats))  # 去重

    def show_img(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.coco_root, self.data_type, img_info['file_name'])

        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im);
        plt.axis('off')

        # 获取该图像对应的anns的Id
        annIds = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(annIds)
        self.coco.showAnns(anns)
        plt.show()


def get_img_list(root):
    """
    读取portrait2000的图片路径和标签路径
    """
    file_list = os.listdir(root)
    file_list = list(filter(lambda x: x.endswith("_matte.png"), file_list))
    label_lst = [os.path.join(root, name) for name in file_list]
    img_lst = [string.replace("_matte.png", ".png") for string in label_lst]
    data_lst = [(path_img, path_label) for path_img, path_label in zip(img_lst, label_lst)]
    return data_lst


def fusion(fore_path, mask_path, back_path):
    """融合图片
    """
    raw_img = cv2.imread(fore_path)
    mask_img = cv2.imread(mask_path) / 255  # 0-1
    back_img = cv2.imread(back_path)

    # 获得抑制背景的前景图片，将像素值限制在0-255之间
    fore_img = np.clip(raw_img * mask_img, a_min=0, a_max=255).astype(np.uint8)

    h, w, c = fore_img.shape
    back_img = cv2.resize(back_img, (w, h))  # 背景图片resize到和前景一样大

    # 获得抑制前景的背景图片，将像素值限制在0-255之间
    result = np.clip(fore_img * mask_img + back_img * (1 - mask_img), a_min=0, a_max=255).astype(np.uint8)

    return result


def gen_img(img_list, coco_genertor, out_dir, img_num=100):
    """生成融合的图片
    
    args:
        img_list: portrait2000的 (人像图片路径，标签图片路径) 列表
        coco_genertor: CocoImg实例 用于获取coco数据集图片路径
        out_dir: 输出的目录
        img_num: 生成数据集的数据量
    """
    for i in range(img_num):
        fore_path, mask_path = random.choice(img_list)  # 随机选择一张前景
        # fore_path, mask_path = img_list[0]  # 调试用，仅用1张前景生成多张图
        _, back_path = random.choice(coco_genertor)  # 随机选择一张背景
        fusion_img = fusion(fore_path, mask_path, back_path)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img_name = "{0:08d}.png".format(i)
        msk_name = "{0:08d}_matte.png".format(i)
        img_path = os.path.join(out_dir, img_name)
        mask_path_dst = os.path.join(out_dir, msk_name)
        cv2.imwrite(img_path, fusion_img)
        shutil.copyfile(mask_path, mask_path_dst)
        print(f"{i}/{img_num}")


if __name__ == '__main__':

    coco_root = r"D:\Datasets\coco"
    data_type = "val2017"  # train2017::118287 张图片, val2017::5000
    super_cats_in = ["outdoor", "indoor"]
    super_cats_out = ["person"]
    
    # step1：创建coco数据集生成器
    coco_genertor = CocoImg(coco_root, data_type, super_cats_in, super_cats_out)
    # for i in range(5):
    #     img_id, img_path = coco_genertor[i]
    #     coco_genertor.show_img(img_id)

    portarit_root = r"D:\Datasets\Portrait-dataset-2000\dataset\training"
    # step2: 获取portrait imglist
    img_list = get_img_list(portarit_root)

    # step3：执行生成
    img_num = 8  # 生成数据集的数据量
    out_dir = r"D:\Datasets\Portrait-dataset-2000\data_aug_{}".format(img_num)
    gen_img(img_list, coco_genertor, out_dir, img_num=img_num)
