# 一些可变的参数配置
import time
from easydict import EasyDict
from torchvision import transforms


cfg = EasyDict()
cfg.train_dir = r"e:\data\cifar10\cifar10_train"
cfg.valid_dir = r"e:\data\cifar10\cifar10_test"
cfg.batch_size = 128
cfg.num_workers = 2  # 子进程数量
cfg.model_name = 'resnet20_cifar'
cfg.num_cls = 10
cfg.pb_sampling = True  # 是否采用渐进式平衡采样

cfg.max_epoch = 200
cfg.lr0 = 0.01  # 初始学习率
cfg.momentum = 0.9  # 动量因子
cfg.weight_decay = 1e-4  # 权重衰减因子 防止过拟合
cfg.milestones = [100, 150]
cfg.decay_factor = 0.1

cfg.mixup = 0  # 触发mixup概率  
cfg.mixup_alpha = 0.2  # <1 凹  >1凸

cfg.label_smoothing = False
cfg.smoothing_factor = 0.001  # 1-0.001

cfg.log_interval = 10  # iter
time_str = time.strftime("%Y%m%d-%H%M")
cfg.output_dir = f"ouputs/{time_str}_cifar"
cfg.log_path = cfg.output_dir + "/log.txt"

# 数据相关
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
cfg.train_transform = transforms.Compose([
    transforms.Resize(32),  # (256, 256)区别  256：短边保持256  1920x1080 [1080->256 1920*(1080/256)]
    transforms.RandomCrop(32),  # 模型最终的输入大小[224, 224]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # 1)0-225 -> 0-1 float  2)HWC -> CHW  -> BCHW
    transforms.Normalize(norm_mean, norm_std)  # 减去均值 除以方差
])

cfg.valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # 0-225 -> 0-1 float HWC-> CHW   BCHW
    transforms.Normalize(norm_mean, norm_std)  # 减去均值 除以方差
])