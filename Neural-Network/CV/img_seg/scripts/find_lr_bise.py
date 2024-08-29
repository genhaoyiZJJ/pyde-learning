"""寻找最合适的lr0
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from datasets.portrait_dataset import PortraitDataset2000
from models import build_model
from utils.common import *


def find_lr(cfg):
    setup_seed(42)  # 先固定随机种子

    # 数据相关
    # 实例化dataset(train valid)
    train_dataset = PortraitDataset2000(cfg.data_root, is_train=True)

    # 组装dataloader
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # 实例化网络模型
    model = build_model(cfg.model_name, cfg.num_cls, pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    # 优化器相关
    # loss函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器实例化
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # 寻找学习率
    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=200)
    lr_finder.plot()


if __name__ == '__main__':

    from configs.config_camvid import cfg
    cfg.model_name = 'deeplabv3+'
    find_lr(cfg)