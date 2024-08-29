# 肖像分割一些可变的参数配置
import time
from easydict import EasyDict
import albumentations as A


cfg = EasyDict()
cfg.dataset_name = "portrait2000"
cfg.data_root = r"e:\data\Portrait-dataset-2000"
cfg.batch_size = 2
cfg.num_workers = 2
cfg.model_name = 'bisenetv1'
cfg.num_cls = 1  # exclude background
cfg.input_size = (512, 512)

cfg.max_epoch = 150

cfg.loss_name = 'bfocal'
cfg.momentum = 0.9
cfg.weight_decay = 5e-4

cfg.scheduler_name = 'cos_warmup'
cfg.lr0 = 0.001
cfg.final_lr = 1e-5
cfg.warmup_epochs = 5
cfg.warmup_init_lr = 1e-4

cfg.milestones = [75, 130]  # [25, 45]
cfg.decay_factor = 0.1

cfg.grad_clip = False
cfg.clip_value = 0.3
cfg.hist_grad = False

cfg.log_interval = 50
time_str = time.strftime("%Y%m%d-%H%M")
cfg.output_dir = f"ouputs/{time_str}_portrait"
cfg.log_path = cfg.output_dir + "/log.txt"


norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
cfg.train_transform = A.Compose([
    A.Resize(width=cfg.input_size[1], height=cfg.input_size[0]),
    A.Normalize(norm_mean, norm_std),
])

cfg.valid_transform = A.Compose([
    A.Resize(width=cfg.input_size[1], height=cfg.input_size[0]),
    A.Normalize(norm_mean, norm_std),
])