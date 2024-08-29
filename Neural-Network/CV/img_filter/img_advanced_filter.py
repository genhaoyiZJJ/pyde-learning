import os
import re
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np

import matplotlib.pyplot as plt


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size(0), x.size(1), -1, 1), dim=2, keepdim=True)


class FeatureExtractor(nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 排除掉最后两层
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        # 冻结模型
        for p in self.features.parameters():
            p.requires_grad = False
        # 检测是否有GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        self.eval()
        
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def forward(self, x):
        x = self.transforms(x).unsqueeze(0).to(self.device)
        x = self.features(x)  # 1*2048*7*7
        features_mean = F.adaptive_avg_pool2d(x, 1)  # 1*2048*1*1
        features_std = global_std_pool2d(x)  # 1*2048*1*1
        return features_mean, features_std


# 提取图像特征
def get_img_feature(model, img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    feature = model(img)
    return feature


# t-SNE降维
def do_tsne(data, random_state=0):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=random_state)
    return tsne.fit_transform(data), tsne


# 绘制数据图像
def plot_embedding(data, type=None, text=None, title="", colors=None):
    if type is None:
        type = np.zeros_like(data[:, 0])
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if text is not None:
            plt.text(data[i, 0], data[i, 1], str(text[i]),
                     color=plt.cm.Set1((type[i] + 1) / 10.) if colors is None else colors[type[i]],
                     fontdict={'weight': 'bold', 'size': 8})
        else:
            plt.scatter(data[i, 0], data[i, 1], s=3,
                        color=plt.cm.Set1((type[i] + 1) / 10.) if colors is None else colors[type[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig


if __name__ == '__main__':
    root_dir = r"e:\cv_projects\Image-Downloader-1.1.1\download_images\dog"
    remove_dir = root_dir + "/remove"
    os.makedirs(remove_dir, exist_ok=True)

    feature_cache = "feature.npy"
    if os.path.exists(feature_cache):
        feature_list = np.load(feature_cache)
        name_list = []
        for img_name in sorted(os.listdir(root_dir)):
            img_path = root_dir + "/" + img_name
            if os.path.isdir(img_path):
                continue
            idx_name = img_name.rsplit('.')[0].split('_')[1]
            name_list.append(idx_name)

    else:
        # 模型初始化
        model = FeatureExtractor()
        # 提取图像特征
        feature_list = []
        name_list = []
        for img_name in sorted(os.listdir(root_dir)):
            img_path = root_dir + "/" + img_name
            if os.path.isdir(img_path):
                continue

            mean, std = get_img_feature(model, img_path)
            mean = mean.cpu().numpy().reshape(-1)
            std = std.cpu().numpy().reshape(-1)
            feature = np.concatenate((mean, std), 0)  # 4096
            print(feature.shape)
            feature_list.append(feature)
            idx_name = img_name.rsplit('.')[0].split('_')[1]
            name_list.append(idx_name)
        feature_list = np.array(feature_list)
        np.save(feature_cache, feature_list)

    # 特征绘图
    feature_list_tsne, _ = do_tsne(feature_list)
    plot_embedding(feature_list_tsne, text=name_list, title="t-SNE")