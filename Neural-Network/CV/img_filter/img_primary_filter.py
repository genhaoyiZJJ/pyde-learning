import os
import cv2
import shutil
import numpy as np


# 检测输入图像是否需要
def check_img(img_path: str):
    suffix = img_path.rsplit('.')[-1]
    if suffix not in ['jpeg', 'jpg', 'png', 'webp']:
        return False

    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    if img is None:
        return False

    # file info
    file_size = os.path.getsize(img_path)
    img_height, img_width = img.shape[:2]
    if file_size < 10 * 1024 or img_width < 256 or img_height < 256:
        return False

    # image basic feature
    img_dy = img[:img_height-1] - img[1:]
    img_dx = img[:, :img_width-1] - img[:, 1:]
    img_gradient = np.mean(np.abs(img_dx)) + np.mean(np.abs(img_dy))
    print(img_path, "img_gradient =", img_gradient)
    if img_gradient < 50:
        return False

    return True


if __name__ == '__main__':
    root_dir = r"e:\cv_projects\Image-Downloader-1.1.1\download_images\dog"
    remove_dir = root_dir + "/remove"
    os.makedirs(remove_dir, exist_ok=True)
    for img_name in os.listdir(root_dir):
        img_path = root_dir + "/" + img_name
        if os.path.isdir(img_path):
            continue

        if not check_img(img_path):
            shutil.move(img_path, remove_dir)
