import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        # 因CE中取了log，所以要exp回来，就得到概率。因为输入并不是概率，CEloss中自带softmax转为概率形式
        pt = torch.exp(-logpt)
        loss_tmp = ((1-pt)**self.gamma) * logpt

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss


if __name__ == "__main__":

    target = torch.tensor([1], dtype=torch.long)
    gamma_lst = [0, 0.5, 1, 2, 5]
    loss_dict = {}
    for gamma in gamma_lst:
        focal_loss_func = FocalLoss(gamma=gamma)
        loss_dict.setdefault(gamma, [])

        for i in np.linspace(0, 10.0, num=30):
            outputs = torch.tensor([[5, i]], dtype=torch.float)  # 制造不同概率的输出
            prob = F.softmax(outputs, dim=1)  # 由于pytorch的CE自带softmax，因此想要知道具体预测概率，需要自己softmax
            loss = focal_loss_func(outputs, target)
            loss_dict[gamma].append((prob[0, 1].item(), loss.item()))

    for gamma, value in loss_dict.items():
        x_prob = [prob for prob, loss in value]
        y_loss = [loss for prob, loss in value]
        plt.plot(x_prob, y_loss, label="γ="+str(gamma))

    plt.title("Focal Loss")
    plt.xlabel("probability of ground truth class")
    plt.xlim(0, 1)
    plt.ylabel("loss")
    plt.ylim(0, 5)
    plt.legend()
    plt.show()


