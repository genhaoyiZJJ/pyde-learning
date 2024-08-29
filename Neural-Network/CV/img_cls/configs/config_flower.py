# 一些可变的参数配置
import time
from easydict import EasyDict
from torchvision import transforms


cfg = EasyDict()
cfg.train_dir = r"E:\data\flowers_data\train"
cfg.valid_dir = r"E:\data\flowers_data\valid"
cfg.batch_size = 64
cfg.num_workers = 2
cfg.model_name = 'resnet18'
cfg.num_cls = 102

cfg.max_epoch = 100
cfg.lr0 = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.milestones = [25, 35]
cfg.decay_factor = 0.1

cfg.mixup = 0
cfg.mixup_alpha = 0.2

cfg.label_smoothing = True
cfg.smoothing_factor = 0.001

cfg.log_interval = 10  # iter
time_str = time.strftime("%Y%m%d-%H%M")
cfg.output_dir = f"ouputs/{time_str}"
cfg.log_path = cfg.output_dir + "/log.txt"

# 数据相关
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
cfg.train_transform = transforms.Compose([
    transforms.Resize(256),  # (256, 256)区别  256：短边保持256  1920x1080 [1080->256 1920*(1080/256)]
    transforms.RandomCrop(224),  # 模型最终的输入大小[224, 224]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # 1)0-225 -> 0-1 float  2)HWC -> CHW  -> BCHW
    transforms.Normalize(norm_mean, norm_std)  # 减去均值 除以方差
])

cfg.valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 0-225 -> 0-1 float HWC-> CHW   BCHW
    transforms.Normalize(norm_mean, norm_std)  # 减去均值 除以方差
])