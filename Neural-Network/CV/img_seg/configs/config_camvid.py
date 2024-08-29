# 一些可变的参数配置
import time
from easydict import EasyDict
from torchvision import transforms


cfg = EasyDict()
cfg.dataset_name = "camvid"
cfg.data_root = r"e:\data\CamVid"
cfg.batch_size = 2
cfg.num_workers = 2
cfg.model_name = 'deeplabv3+'
cfg.num_cls = 12

cfg.max_epoch = 150

cfg.loss_name = 'ce'
cfg.lr0 = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.milestones = [75, 130]
cfg.decay_factor = 0.1

cfg.grad_clip = True
cfg.clip_value = 0.3
cfg.hist_grad = True

cfg.log_interval = 10
time_str = time.strftime("%Y%m%d-%H%M")
cfg.output_dir = f"ouputs/{time_str}_camvid"
cfg.log_path = cfg.output_dir + "/log.txt"
