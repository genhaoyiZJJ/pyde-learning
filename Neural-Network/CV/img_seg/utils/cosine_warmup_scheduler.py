
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.models import resnet18
from math import pi, cos
from torch.optim.optimizer import Optimizer


class CosineWarmupLr(object):
    """Cosine lr decay function with warmup.
    Lr warmup is proposed by `
        Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by `
        Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_per_epoch (int): batches of one epoch.
        max_epochs (int): max_epochs to train.
        base_lr (float): init lr.
        final_lr (float): minimum(final) lr.
        warmup_epochs (int): warmup max_epochs before cosine decay.
        warmup_init_lr (float): warmup starting lr.
        last_iter (int): init iteration.
    Attributes:
        niters (int): number of iterations of all max_epochs.
        warmup_iters (int): number of iterations of all warmup max_epochs.
    """
    def __init__(self, optimizer, iter_per_epoch, max_epochs, base_lr, final_lr=0,
                 warmup_epochs=0, warmup_init_lr=0, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.baselr = base_lr
        self.learning_rate = base_lr
        self.niters = max_epochs * iter_per_epoch
        self.targetlr = final_lr
        self.warmup_iters = iter_per_epoch * warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        self.last_iter = last_iter
        self.step()

    def get_lr(self):
        if self.last_iter < self.warmup_iters:  # warmup
            self.learning_rate = self.warmup_init_lr + \
                (self.baselr - self.warmup_init_lr) * self.last_iter / self.warmup_iters
        else: 
            self.learning_rate = self.targetlr + (self.baselr - self.targetlr) * \
                (1 + cos(pi * (self.last_iter - self.warmup_iters) /
                         (self.niters - self.warmup_iters))) / 2

    def step(self, iteration=None):
        """Update status of lr.
        Args:
            iteration(int, optional): now training iteration of all max_epochs.
                Normally need not to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate


if __name__ == '__main__':
    
    warmup_epochs = 6  # 
    max_epoch = 120  # 共有120个epoch，则用于cosine rate的一共有100个epoch
    lr_init = 0.1
    lr_final = 1e-5
    lr_warmup_init = 0.

    iter_per_epoch = 100

    model = resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    scheduler = CosineWarmupLr(optimizer, iter_per_epoch=1, max_epochs=max_epoch, base_lr=0.1,
                               final_lr=lr_final, warmup_epochs=warmup_epochs, warmup_init_lr=lr_warmup_init)
    index = 0
    x = []
    y = []
    for epoch in range(max_epoch):
        for iter in range(iter_per_epoch):
            lr_c = optimizer.param_groups[0]['lr']
            print(lr_c)
            index += 1
        y.append(lr_c)
        x.append(epoch)
        scheduler.step()

    plt.figure(figsize=(10, 8))
    plt.xlabel('epoch')
    plt.ylabel('cosine rate')
    plt.plot(x, y, color='r', linewidth=2.0, label='cosine rate')
    plt.legend(loc='best')
    plt.show()

