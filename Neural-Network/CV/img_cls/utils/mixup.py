import cv2
import numpy as np
import torch


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.):
    """mixup一个batch

    Args:
        x (torch.Tensor): b*c*h*w
        y (torch.Tensor): b*1
        alpha (float, optional): beta分布的alpha值. Defaults to 1.
    """
    # 去概率
    lamb = np.random.beta(alpha, alpha)
    
    # 打乱batch 获得id  [0, 1, 2, ...]
    index = torch.randperm(x.shape[0]).to(x.device)
    
    # 通过id获得打乱后的batch
    x1, y1 = x, y
    x2, y2 = x[index], y[index]
    
    # mixup
    x_mixup = (lamb * x1 + (1-lamb) * x2)
    return x_mixup, y1, y2, lamb


if __name__ == "__main__":
    
    # 读取两张图片

    img1 = cv2.imread(r"e:\data\flowers_data\reorder\14\image_06355.jpg")
    img2 = cv2.imread(r"e:\data\flowers_data\reorder\98\image_07862.jpg")

    # resize固定尺寸
    img1 = cv2.resize(img1, (224, 224))  # [0-225]  uint8
    img2 = cv2.resize(img2, (224, 224))

    # 取概率值
    alpha = 1
    lamb = np.random.beta(alpha, alpha)
    print(lamb)

    # mixup
    img_mixup = (lamb * img1 + (1-lamb) * img2).astype(np.uint8)

    winname = f'lamb: {lamb:0.3f}'
    cv2.namedWindow(winname, cv2.WINDOW_FREERATIO)
    cv2.imshow(winname, img_mixup)
    cv2.waitKey()