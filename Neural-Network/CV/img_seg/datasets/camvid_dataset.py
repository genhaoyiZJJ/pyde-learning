import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CamvidDataset(Dataset):
    names = ('Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 
             'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Unlabelled')
    def __init__(self, data_root, is_train=True):

        self.image_dir = os.path.join(data_root, 'train' if is_train else 'val')
        self.label_dir = os.path.join(data_root, 'trainannot' if is_train else 'valannot')
        assert os.path.exists(self.image_dir)
        assert os.path.exists(self.label_dir)
        
        self.data_info = []
        # 从硬盘中读取图片和标签路径
        self._get_data_info()
      
        self.transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        img, label = self.data_info[index]

        img = Image.open(img)
        label = Image.open(label)

        img = self.transform_img(img)
        label = torch.from_numpy(np.array(label, dtype='int64'))
        return img, label  # (3, 360, 480)  (360, 480)

    def __len__(self):
        return len(self.data_info)

    def _get_data_info(self):
        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            lbl_path = os.path.join(self.label_dir, img_name)
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                self.data_info.append((img_path, lbl_path))
        assert len(self.data_info)


if __name__ == "__main__":

    camvid_dir = r"e:\data\CamVid"

    crop_size = (360, 480)
    train_data = CamvidDataset(camvid_dir, is_train=True)
    # exit()
    print(len(train_data))
    img_data, img_label = train_data[0]
    print(img_data.shape, img_label.shape)
    
    import cv2
    label2 = cv2.imread(r"e:\data\CamVid\trainannot\0001TP_006690.png", cv2.IMREAD_GRAYSCALE)

    img, label = img_data.numpy(), img_label.numpy()
    img = img.transpose(1, 2, 0)
    
    # 创建一个具有两个子图的画布
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # 在第一个子图上显示第一张图片
    ax1.imshow(img)
    ax1.set_title('Image')

    # 在第二个子图上显示第二张图片
    ax2.imshow(label)
    ax2.set_title('Label')

    ax3.imshow(label2)
    ax3.set_title('Label2')
    # 显示画布
    plt.show()
    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

    # for i, data in enumerate(train_loader):
    #     img_data, img_label = data
    #     print(img_data.shape, img_data.dtype, type(img_data))       # torch.float32
    #     print(img_label.shape, img_label.dtype, type(img_label))     # torch.longint(int64)
