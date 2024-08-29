"""  albumentations用于图像分割的demo
"""

import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
    
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask, cmap ='gray')
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask, cmap ='gray')
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


if __name__ == '__main__':

    image = cv2.imread("00079.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('00079_matte.png', cv2.IMREAD_GRAYSCALE)

    # step1：定义好一系列变换方法
    aug = A.Compose([
        # A.Resize(width=336, height=448),
        # A.Blur(blur_limit=7, p=1),  # 采用随机大小的kernel对图像进行模糊
        # A.ChannelDropout(p=1),  # 随机选择某个通道像素值设置为 0
        # A.ChannelShuffle(p=1),    # 颜色通道随机打乱 rgb --> bgr/brg/rbg/rgb/grb/gbr
        # A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2, p=1),  # 颜色扰动：亮度、对比度、饱和度、色相
        # A.GaussNoise(var_limit=(100, 255), p=1),  # 高斯噪点
        # A.InvertImg(p=1),  # 像素反转 255-pixel_value
        # A.Normalize(max_pixel_value=255.0, p=1.0),
        # A.Affine(scale=(0.5, 1.5), p=1),       # 缩放
        # A.Affine(translate_percent=0.1, p=1),  # 平移
        # A.Affine(rotate=(-20, 20), mode=cv2.BORDER_REFLECT, p=1),  # 旋转 [cv2.BORDER_]
        # A.Affine(shear=(10, 30), mode=cv2.BORDER_REFLECT, p=1),   # 错切
        # A.CoarseDropout(max_holes=20, max_height=20, max_width=20, p=1),   #  对于工业场景，适用。《Improved Regularization of Convolutional Neural Networks with Cutout》
        # A.ElasticTransform(p=1, border_mode=1),  # 弹性变形。alpha越小，sigma越大，产生的偏差越小，和原图越接近
        # A.LongestMaxSize(max_size=500, p=1),    # 依最长边保持比例的缩放
        # A.OneOf([
        #     A.HorizontalFlip(p=1),
        #     A.VerticalFlip(p=1),
        #     A.Sequential([
        #         A.HorizontalFlip(p=1),
        #         A.VerticalFlip(p=1),
        #     ], p=1),
        # ], p=1),
        # A.Sequential([
        #     A.HorizontalFlip(p=1),
        #     A.VerticalFlip(p=1),
        # ], p=1),
    ])

    # step2：给该变换出入源数据（通常在Dataset的__getitem__中使用）
    augmented = aug(image=image, mask=mask)
    # step3：获取变换后的数据 （通常在Dataset的__getitem__中使用）
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    # 观察效果
    print("raw: ", image.shape, mask.shape)
    print("aug: ", image_aug.shape, mask_aug.shape)
    visualize(image_aug, mask_aug, original_image=image, original_mask=mask)