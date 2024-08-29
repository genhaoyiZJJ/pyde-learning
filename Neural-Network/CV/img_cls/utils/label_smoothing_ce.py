from torch import nn
import torch
from torch.nn import functional as F


class CE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        """交叉熵推理过程

        Args:
            y_pred (torch.Tensor): b*num_cls
            y_label (torch.Tensor): b*1

        Returns:
            torch.Tensor: loss
        """
        # softmax -> log
        log_prod = F.log_softmax(y_pred, dim=-1)
        
        # one-hot label
        one_hot = torch.zeros_like(log_prod)
        for i in range(len(one_hot)):
            t =y_label[i]
            one_hot[i][t] = 1
        
        # x + 
        loss = (-(log_prod * one_hot).sum(dim=-1)).mean(dim=0)
        return loss
    

class CE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        """交叉熵推理过程

        Args:
            y_pred (torch.Tensor): b*num_cls
            y_label (torch.Tensor): b*1

        Returns:
            torch.Tensor: loss
        """
        # softmax -> log
        log_prod = F.log_softmax(y_pred, dim=-1)
        
        # one-hot label
        one_hot = torch.zeros_like(log_prod)
        for i in range(len(one_hot)):
            t =y_label[i]
            one_hot[i][t] = 1
        
        # x + 
        loss = (-(log_prod * one_hot).sum(dim=-1)).mean(dim=0)
        return loss


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.01):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, y_pred: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        """label_smoothing交叉熵推理过程

        Args:
            y_pred (torch.Tensor): b*num_cls
            y_label (torch.Tensor): b*1

        Returns:
            torch.Tensor: loss
        """
        # softmax -> log
        log_prod = F.log_softmax(y_pred, dim=-1)

        # label smoothing label
        # smooth_label = torch.ones_like(log_prod) * self.smoothing / (len(y_label) - 1)
        smooth_label = torch.ones_like(log_prod) * self.smoothing / (y_pred.size(-1) - 1)
        for i in range(len(smooth_label)):
            t =y_label[i]
            smooth_label[i][t] = 1 - self.smoothing
        
        # x + 
        loss = (-(log_prod * smooth_label).sum(dim=-1)).mean(dim=0)
        return loss
    

if __name__ == "__main__":
    y_pred = torch.tensor([[0.05, 0.1, 0.4], [2, 4, 5]])
    y_label = torch.tensor([1, 2])
    
    loss_fn1 = CE()
    loss1 = loss_fn1(y_pred, y_label)
    print(f"loss1: {loss1}")
    
    loss_fn2 = nn.CrossEntropyLoss()
    loss2 = loss_fn2(y_pred, y_label)
    print(f"loss2: {loss2}")
    
    loss_fn3 = LabelSmoothingCE(smoothing=0.001)
    loss3 = loss_fn3(y_pred, y_label)
    print(f"loss3: {loss3}")
    