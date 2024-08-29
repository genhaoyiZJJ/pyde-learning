import torch
import torch.nn as nn


def build_loss(name, cfg):
    if name == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    elif name == 'bce':
        pos_weight = torch.tensor(cfg['bce_pos_weight']) if 'bce_pos_weight' in cfg else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif name == "dice":
        from .dice_loss import DiceLoss

        loss_fn = DiceLoss()
    elif name == "bce+dice":
        from .bce_dice_loss import BCEDiceLoss

        loss_fn = BCEDiceLoss()
    elif name == "bfocal":
        from .focal_loss_binary import BinaryFocalLossWithLogits

        kwargs = {"alpha": cfg.get("focal_alpha", 0.25), 
                  "gamma": cfg.get("focal_gamma", 2), 
                  "reduction": 'mean'}
        loss_fn = BinaryFocalLossWithLogits(**kwargs)
        
    else:
        raise Exception(f'{name} loss function is not supported')

    return loss_fn