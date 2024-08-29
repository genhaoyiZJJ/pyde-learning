# -*- encoding: utf-8 -*-
"""
@File   : model_trainer.py
@Desc   : 通用模型训练类
@Time   : 2023/06/22
@Author : fan72
"""
from typing import List
import torch
import numpy as np
from torch.nn.utils import clip_grad_value_
from utils.evalution_segmentaion import eval_semantic_segmentation


class ModelTrainer(object):

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, 
                        scheduler, epoch_idx, device, logger, cfg,
                        grad_list_epoch: List):
        model.train()

        class_num = cfg.num_cls + 1 if cfg.num_cls == 1 else cfg.num_cls
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        train_acc = []
        train_miou = []
        grad_list = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()

            # 计算loss
            if isinstance(outputs, (tuple, list)):
                loss = sum([loss_f(output.cpu(), labels.cpu()) for output in outputs])
            else:
                loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()

            # 梯度裁剪
            if cfg.grad_clip:
                clip_grad_value_(model.parameters(), cfg.clip_value)

            # 统计最大梯度值
            if cfg.hist_grad:
                tmp = []
                for p in model.parameters():
                    tmp.append(torch.max(p.grad.abs()).cpu().numpy())
                grad_list.append(max(tmp).flatten()[0])

            optimizer.step()

            # 评估
            # pred_label
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if labels.dim() == 4:  # 多标签预测 || 单类二值预测

                pred_label = outputs.sigmoid().squeeze(1)
                pred_label = (pred_label > 0.5).long().cpu().numpy()
                true_label = labels.squeeze(1).round().long()
                true_label = true_label.cpu().numpy()
                    
            else:
                pred_label = outputs.max(dim=1)[1].cpu().numpy()  # (bs, 360, 480)
                # true_label
                true_label = labels.data.cpu().numpy()

            eval_metrix = eval_semantic_segmentation(pred_label, true_label, class_num)  # P:(bs, h, w)  T:(bs, h, w)
            train_acc.append(eval_metrix['mean_class_accuracy'])
            train_miou.append(eval_metrix['iou'][1] if cfg.num_cls == 1 else eval_metrix['miou'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss.item())

            # 间隔 log_interval 个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info('|Epoch[{:0>3}/{:0>3}]||Iteration[{:0>3}/{:0>3}]|batch_loss: {:.4f}||mIoU {:.4f}|'.format(
                    epoch_idx, cfg.max_epoch, i + 1, len(data_loader), loss.item(), eval_metrix['miou']))

        grad_list_epoch.extend(grad_list)

        # 每个epoch结束后更新一次学习率
        scheduler.step()

        loss_mean = np.mean(loss_sigma)
        acc_mean = np.mean(train_acc)
        miou_mean = np.mean(train_miou)
        return loss_mean, acc_mean, conf_mat, miou_mean

    @staticmethod
    def valid_one_epoch(data_loader, model, loss_f, device, cfg):
        model.eval()

        class_num = cfg.num_cls + 1 if cfg.num_cls == 1 else cfg.num_cls
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        valid_acc = []
        valid_miou = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 不进行梯度计算
            with torch.no_grad():
                outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                loss = sum([loss_f(output.cpu(), labels.cpu()) for output in outputs])
            else:
                loss = loss_f(outputs.cpu(), labels.cpu())

            # 统计loss
            loss_sigma.append(loss.item())

            # 评估
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if labels.dim() == 4:  # 多标签预测

                pred_label = outputs.sigmoid().squeeze(1)
                pred_label = (pred_label > 0.5).long().cpu().numpy()
                true_label = labels.squeeze(1).round().long()
                true_label = true_label.cpu().numpy()
                    
            else:
                pred_label = outputs.max(dim=1)[1].cpu().numpy()  # (bs, 360, 480)
                # true_label
                true_label = labels.data.cpu().numpy()

            eval_metrix = eval_semantic_segmentation(pred_label, true_label, class_num)
            valid_acc.append(eval_metrix['mean_class_accuracy'])
            valid_miou.append(eval_metrix['iou'][1] if cfg.num_cls == 1 else eval_metrix['miou'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss.item())

        loss_mean = np.mean(loss_sigma)
        acc_mean = np.mean(valid_acc)
        miou_mean = np.mean(valid_miou)

        return loss_mean, acc_mean, conf_mat, miou_mean


