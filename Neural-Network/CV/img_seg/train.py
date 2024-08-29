"""分割训练脚本
"""
import argparse
import time
import os
import shutil
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import build_dataset
from losses import build_loss
from models import build_model
from utils.model_trainer import ModelTrainer
from utils.common import *
from utils.evalution_segmentaion import calc_semantic_segmentation_iou


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=None, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=None, type=int, help='training batch size')
    parser.add_argument('--max_epoch', type=int, default=None, help='number of epoch')
    args = parser.parse_args()
    return args


def train(cfg):
    setup_seed(42)  # 先固定随机种子

    logger = setup_logger(cfg.log_path, 'w')
    # 参数配置
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 数据相关
    # 实例化dataset(train valid)
    train_dataset = build_dataset(cfg.dataset_name, cfg, is_train=True)

    # valid
    valid_dataset = build_dataset(cfg.dataset_name, cfg, is_train=False)

    # 组装dataloader
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 实例化网络模型
    model = build_model(cfg.model_name, cfg.num_cls, pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    # 优化器相关
    # loss函数
    loss_fn = build_loss(cfg.loss_name, cfg)

    # 优化器实例化
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr0, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # 学习率下降策略的实例化
    if cfg.scheduler_name == "cos_warmup":
        from utils.cosine_warmup_scheduler import CosineWarmupLr
        lr_scheduler = CosineWarmupLr(optimizer, iter_per_epoch=1, max_epochs=cfg.max_epoch,
                                      base_lr=cfg.lr0, final_lr=cfg.final_lr, 
                                      warmup_epochs=cfg.warmup_epochs, warmup_init_lr=cfg.warmup_init_lr)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.decay_factor)

    logger.info(
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
            cfg, loss_fn, lr_scheduler, optimizer, cfg.model_name
        )
    )
    # loop
    logger.info("start train...")
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    miou_rec = {"train": [], "valid": []}
    best_miou, best_epoch = 0, 0
    grad_list_epoch = []
    t_start = time.time()
    for epoch in range(cfg.max_epoch):
        # 一次epoch的训练
        # 按batch形式取数据
        # 前向传播
        # 计算Loss
        # 反向传播计算梯度
        # 更新权重
        # 统计Loss 评价指标
        loss_train, acc_train, conf_mat_train, miou_train = ModelTrainer.train_one_epoch(
            train_loader, model, 
            loss_f=loss_fn, 
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch_idx=epoch,
            device=device,
            logger=logger,
            cfg=cfg,
            grad_list_epoch=grad_list_epoch,
        )

        # 一次epoch验证
        # 按batch形式取数据
        # 前向传播
        # 计算Loss
        # 统计Loss 评价结果
        loss_valid, acc_valid, conf_mat_valid, miou_valid = ModelTrainer.valid_one_epoch(
            valid_loader, 
            model, loss_fn, 
            device=device,
            cfg=cfg,
        )

        # 打印训练集和验证集上的指标
        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%}\n"
                    "Train loss:{:.4f} Train miou:{:.4f}\n"
                    "Valid loss:{:.4f} Valid miou:{:.4f}\n"
                    "LR:{}". format(epoch, cfg.max_epoch, acc_train, acc_valid, 
                                    loss_train, miou_train,
                                    loss_valid, miou_valid, 
                                    optimizer.param_groups[0]["lr"]))

        # 保存混淆矩阵图
        show_confMat(conf_mat_train, train_dataset.names, "train", cfg.output_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)
        show_confMat(conf_mat_valid, valid_dataset.names, "valid", cfg.output_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        miou_rec["train"].append(miou_train), miou_rec["valid"].append(miou_valid)
        # 保存loss曲线， acc曲线， miou曲线
        plt_x = np.arange(1, epoch + 2)  # list(range(1, epoch + 2))
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=cfg.output_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=cfg.output_dir)
        plot_line(plt_x, miou_rec["train"], plt_x, miou_rec["valid"], mode="miou", out_dir=cfg.output_dir)

        # 保存模型
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "miou": miou_valid
        }
        torch.save(checkpoint, f"{cfg.output_dir}/last.pth")

        # 保存验证集上表现最好的模型
        if best_miou < miou_valid:
            best_miou, best_epoch = miou_valid, epoch
            shutil.copy(f"{cfg.output_dir}/last.pth", f"{cfg.output_dir}/best.pth")

        # 观察各类别的iou：
        iou_array = calc_semantic_segmentation_iou(conf_mat_valid)
        info = ["{}_iou:{:.2f}".format(n, iou) for n, iou in zip(train_dataset.names, iou_array)]
        logger.info("per class mIoU: \n{}".format("\n".join(info)))

        if cfg.hist_grad:
            grad_png_path = os.path.join(cfg.output_dir, "grad_hist.png")
            max_grad = max(grad_list_epoch)
            logger.info("max grad in {}, is {}".format(grad_list_epoch.index(max_grad), max_grad))
            import matplotlib.pyplot as plt

            plt.hist(grad_list_epoch)
            plt.savefig(grad_png_path)

    t_use = (time.time() - t_start) / 3600
    logger.info(f"Train done, use time {t_use:.3f} hours, best miou: {best_miou:.3f} in :{best_epoch}")


if __name__ == '__main__':
    args = parse_args()
    
    from configs.config_portrait import cfg

    # update cfg
    cfg.lr0 = args.lr if args.lr else cfg.lr0
    cfg.batch_size = args.batch_size if args.batch_size else cfg.batch_size
    cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

    train(cfg)